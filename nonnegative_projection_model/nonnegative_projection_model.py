import abc, warnings, re
import numpy as np
import scipy as sp
import theano
import theano.tensor as T

from scipy.optimize import minimize

##########
## data ##
##########

class Data(object):

    def _load_X(self, fname, obj_filter=lambda obj: not re.match('^(be|become|are)\s', obj)):
        data = np.loadtxt(fname, 
                          dtype=str, 
                          delimiter=',')

        objects = data[1:,0]
        obsfeats = data[0,1:]
        X = data[1:,1:].astype(int)

        obj_filter_vectorized = np.vectorize(obj_filter)

        objects_filtered = obj_filter_vectorized(objects)
        obsfeats_filtered = X.sum(axis=0) > 0

        self.objects = objects[objects_filtered]
        self.obsfeats = obsfeats[obsfeats_filtered]
        self._X = X[objects_filtered][:,obsfeats_filtered]

    def _initialize_ranges(self):
        self._obj_range = range(self.objects.shape[0])
        self._obsfeat_range = range(self.obsfeats.shape[0])

    def get_data(self):
        return self._X

    def get_obj_range(self):
        return self._obj_range

    def get_obsfeat_range(self):
        return self._obsfeat_range


class BatchData(Data):

    def __init__(self, data_fname):
        self._load_X(data_fname)

        self._initialize_obj_counts()
        self._initialize_ranges()

    def _initialize_obj_counts(self):
        self._X_obj_count = self._X.sum(axis=1)

    def get_obj_counts(self):
        return self._X_obj_count

class IncrementalData(Data):

    def __init__(self, data_fname, seed=0):
        np.random.seed(seed)

        self._load_X(data_fname)

        self._initialize_obj_counts()
        self._initialize_ranges()

        self._seen = np.zeros(self._X.shape)
        self._initialize_pair_probs()

    def __iter__(self):
        return self

    def _initialize_obj_counts(self):
        self._X_obj_count = np.zeros(self._X.shape[0])
        self._unseen = self._X.astype(float)

    def _update_pair_probs(self):
        self._pair_probs = (self._unseen / self._unseen.sum()).flatten()

    def _sample_pair(self):
        pair_index = np.random.choice(a=self._pairs_probs.shape[0], 
                                      p=self._pair_probs)

        obj_index = pair_index / self._X.shape[1]
        obsfeat_index = pair_index / self._X.shape[0]
        
        self._unseen[obj_index, obsfeat_index] -= 1
        self._seen[obj_index, obsfeat_index] += 1

        self._update_pair_probs()

        return obj_index, obsfeat_index

    def next(self):
        try:
            assert self._unseen.sum() > 0
        except AssertionError:
            raise StopIteration

        datum = np.zeros(self._X.shape)
        datum[self._sample_pair()] += 1

        return datum


##############
## likelihoods
##############

class Likelihood(object):

    @abc.abstractmethod
    def compute_log_likelihood(self, parameter, **kwargs):
        return

    @abc.abstractmethod
    def get_data(self):
        return

class Prior(object):

    @abc.abstractmethod
    def compute_log_prior(self, parameter, **kwargs):
        return

class PoissonGammaProductLikelihood(Likelihood):

    def __init__(self, data, gamma=1.):
        self.gamma = gamma

        self._initialize_data(data)

    def _initialize_data(self, data):
        self._data = data

        if isinstance(data, BatchData):
            self._X = data.get_data()
            self._X_obj_count = data.get_obj_counts()
            self._obj_range = data.get_obj_range()
            self._obsfeat_range = data.get_obsfeat_range()

        elif isinstance(data, IncrementalData):
            self._X_gen = data

    def link_prior_params(self, D_inference):
        self._D = D_inference.get_param()

    def get_data(self):
        return self._data.get_data()

    def _construct_indices(self, obj, obsfeat):
        obj = self._obj_range if obj == None else obj
        obsfeat = self._obsfeat_range if obsfeat == None else obsfeat

        return obj, obsfeat

    def _compute_ll_d_new(self, d_new, obj, obsfeat):
        log_numer = self._X[obj,obsfeat] * np.log(d_new)

        d_obj = np.insert(np.delete(self._D[obj], 
                                    obsfeat), 
                          obsfeat, 
                          d_new)

        log_denom_X = 1 + self._X_obj_count[obj]
        log_denom_D = np.log(self.gamma + d_obj.sum())
        log_denom = log_denom_X * log_denom_D

        return log_numer - log_denom

    def _compute_ll_D(self):
        log_numer = np.sum(self._X * np.log(self._D), axis=1)

        log_denom_X = 1 + self._X_obj_count[:,None]
        log_denom_D = np.log(self.gamma + self._D.sum(axis=1))
        log_denom = log_denom_X * log_denom_D

        return np.sum(log_numer - log_denom)


    def compute_log_likelihood(self, d_new=None, obj=None, obsfeat=None):
        if d_new == None:
            return self._compute_ll_D()
        else:
            return self._compute_ll_d_new(d_new, obj, obsfeat)


class BetaPosterior(Prior, Likelihood):

    def __init__(self, D_inference):
        self._D = D_inference.get_param()
        D_inference.link_prior(self)

        self._B_inference = None

        self._Z = None
        self._B = None

    def link_prior_params(self, Z_inference=None, B_inference=None):
        ## don't need to save Z_inference
        self._B_inference = B_inference if B_inference != None else self._B_inference

        self._Z = Z_inference.get_param() if Z_inference != None else self._Z
        self._B = B_inference.get_param() if B_inference != None else self._B

        if self._Z != None and self._B != None:
            self._ZB = np.dot(self._Z, self._B)
            self._update_log_likelihood_values()

    def compress_other_matrices(self, latfeats_gt_0):
        self._B_inference.compress(latfeats_gt_0)

    def append_to_other_matrices(self, num_of_new_features):
        self._B_inference.sample_new_features(num_of_new_features)


    def get_data(self):
        return self._D

    def get_param(self, param):
        if param == 'Z':
            return self._Z

        elif param == 'B':
            return self._B

        else:
            raise ValueError('parameter must equal "Z" or "B"')

    def _compute_ll_z_new(self, z_new, obj, latfeat):
        z_obj = np.insert(np.delete(self._Z[obj], 
                                    latfeat), 
                          latfeat, 
                          z_new)

        z_obj_B = np.dot(z_obj, self._B)
        z_obj_b_latfeat = z_new * self._B[latfeat]

        ll = np.log(z_obj_B) + (z_obj_b_latfeat - 1)*np.log(self._D[obj])
            
        return np.sum(ll)

    def _compute_ll_b_new(self, b_new, latfeat, obsfeat):
        b_obsfeat = np.insert(np.delete(self._B[:,obsfeat], 
                                        latfeat), 
                              latfeat, 
                              b_new)

        Zb_obsfeat = np.dot(self._Z, b_obsfeat)
        z_latfeat_b_obsfeat = b_new * self._Z[:,latfeat]

        ll = np.log(Zb_obsfeat) + (z_latfeat_b_obsfeat - 1)*np.log(self._D[:,obsfeat])

        return np.sum(ll)

    def _compute_ll(self, Z=None, B=None):
        Z = self._Z if Z == None else Z
        B = self._B if B == None else B

        ZB = np.dot(Z, B)
        self._log_likelihood = np.log(ZB) + (ZB - 1) * np.log(self._D)

        return np.sum(self._log_likelihood)

    def compute_log_likelihood(self, Z=None, B=None, z_new=None, b_new=None, 
                               obj=None, latfeat=None, obsfeat=None):
        if z_new != None:
            return self._compute_ll_z_new(z_new, obj, latfeat)

        elif b_new != None:
            return self._compute_ll_b_new(b_new, latfeat, obsfeat)

        elif B!= None:
            ## this would not be necessary if we weren't
            ## doing optimization for B since it could be
            ## done using self._B
            return self._compute_ll(B=B)

        elif Z!= None:
            ## this would not be necessary if we weren't
            ## doing optimization for B since it could be
            ## done using self._B
            return self._compute_ll(Z=Z)

        else:
            self._update_log_likelihood_values()
            return self._log_likelihood_values[obj, obsfeat]

    def compute_log_likelihood_jacobian(self, B):
        one_over_ZB = 1. / np.dot(self._Z, B)
        log_D = np.log(self._D)

        inner_sum = (one_over_ZB + log_D).sum(axis=1)
        outer_sum = (self._Z * inner_sum[:,None]).sum(axis=0)

        return B * outer_sum[:,None]

    def _update_log_likelihood_values(self):
        self._log_likelihood_values = np.log(self._ZB) + (self._ZB - 1) * np.log(self._D)

    def _update_log_prior_values(self):
        self._log_prior_values = (self._ZB - 1) * np.log(self._D)

    def compute_log_prior(self, D=None, d_new=None, obj=None, obsfeat=None):
        if d_new != None:
            return (self._ZB[obj, obsfeat] - 1) * np.log(d_new)

        elif D != None:
            ## this would not be necessary if we weren't
            ## doing optimization for D since it could be
            ## done using self._D or self._log_prior_values
            lp = (self._ZB - 1) * np.log(D)

            return np.sum(lp)

        else:
            self._update_log_prior_values()
            return self._log_prior_values[obj, obsfeat]

###########
## samplers
###########

class Sampler(object):
    
    @abc.abstractmethod
    def get_param(self):
        return 

    @abc.abstractmethod
    def sample(self):
        return 


class DSampler(Sampler):
    
    def __init__(self, D, likelihood, proposal_bandwidth):
        self._initialize_D(D, likelihood)
        self._initialize_ranges()
        self._initialize_proposer(proposal_bandwidth)

    def _initialize_D(self, initial_D, likelihood):
        self._D = initial_D
        self._log_likelihood = likelihood.compute_log_likelihood

        likelihood.link_prior_params(D_inference=self)

    def _initialize_ranges(self):
        self._obj_range = range(self._D.shape[0])
        self._obsfeat_range = range(self._D.shape[1])

    def _initialize_proposer(self, proposal_bandwidth):
        ## find point-wise min of D and 1-D
        D_minimum = np.minimum(self._D, 1-self._D)

        ## use min to precompute proposal window and transition probs
        self._proposal_bandwidth = proposal_bandwidth
        self._proposal_window = D_minimum / proposal_bandwidth
        self._log_proposal_prob_values = np.log(D_minimum)

    def get_param(self):
        return self._D

    def link_prior(self, prior):
        self._log_prior = prior.compute_log_prior

    def _compute_lp_d_new(self, d_new, obj, obsfeat):
        ll =  self._log_likelihood(d_new=d_new, 
                                   obj=obj, 
                                   obsfeat=obsfeat)
        lp = self._log_prior(d_new=d_new, 
                             obj=obj, 
                             obsfeat=obsfeat)

        return ll + lp

    def _compute_lp_D(self, D):
        ll =  self._log_likelihood()
        lp = self._log_prior(D=D)            

        return ll + lp            

    def _compute_log_posterior(self, D=None, d_new=None, obj=None, obsfeat=None):
        if d_new != None:
            return self._compute_lp_d_new(d_new, obj, obsfeat)

        elif D != None:
            return self._compute_lp_D(D)

        else:
            return self._log_posterior_values[obj, obsfeat]

    def _proposer(self, obj, obsfeat):
        window_size = self._proposal_window[obj, obsfeat]
        
        return np.random.uniform(low=self._D[obj,obsfeat]-window_size, 
                                 high=self._D[obj,obsfeat]+window_size)

    def _log_proposal_prob(self, obj, obsfeat, proposal=None):
        if proposal == None:
            return self._log_proposal_prob_values[obj, obsfeat]

        else:
            return np.log(np.min([proposal, 1-proposal]))

    def _acceptance_prob(self, proposal, obj, obsfeat):
        current_log_post_val = self._compute_log_posterior(obj=obj, obsfeat=obsfeat) 
        proposal_log_post_val = self._compute_log_posterior(d_new=proposal, obj=obj, obsfeat=obsfeat) 

        log_post_ratio = current_log_post_val - proposal_log_post_val 

        ## transition probabilities are confusing:
        ## the first is really propto log(p(new->old))
        ## and the second is really propto log(p(old->new))
        ## without constants
        current_log_prop_val = self._log_proposal_prob(obj, obsfeat) 
        proposal_log_prop_val = self._log_proposal_prob(obj, obsfeat, proposal) 

        log_proposal_ratio = current_log_prop_val - proposal_log_prop_val

        return log_post_ratio + log_proposal_ratio

    def _update_D(proposal, obj, obsfeat):
        self._D[obj, obsfeat] = proposal

        d_obj_obsfeat_min = np.min([proposal, 1-proposal])
        self._proposal_window[obj, obsfeat] = d_obj_obsfeat_min / self._proposal_bandwidth
        self._log_proposal_prob_values[obj, obsfeat] = np.log(d_obj_obsfeat_min)

    def _sample(self, obj, obsfeat):
        proposal = self._proposer(obj, obsfeat)

        acceptance_log_prob = self._acceptance_prob(proposal, obj, obsfeat)
        acceptance_log_prob = np.min([0, acceptance_log_prob])

        accept = np.log(np.random.uniform(low=0., high=1.)) < acceptance_log_prob

        if accept:
            self._update_D(proposal, obj, obsfeat)

    def _update_log_posterior_values(self):
        log_prior_vals = self._log_prior(self._obj_range, self._obsfeat_range)
        log_like_vals = self._log_likelihood()

        self._log_posterior_values = log_prior_vals + log_like_vals

    def sample(self):
        self._update_log_posterior_values()

        for obj in self._obj_range:
            for obsfeat in self._obsfeat_range:
                self._sample(obj, obsfeat)


class ZSampler(Sampler):

    """
    Samples verb-by-latent binary features using Gibbs sampling

    Attributes:
      alpha
    """
    
    def __init__(self, Z, likelihood, alpha, beta=1., nonparametric=False):
        """
        Initialize the verb-by-latent binary feature sampler

        Args:
          likelihood (Likelihood): computes p(D|B,Z) and implements interfaces in ABC Likelihood
          alpha (float): positive real number parameterizing Beta(alpha, 1) or IBP(alpha)
          num_of_latfeats (int): number of latent binary features; 
                                 if 0, the sampler is nonparametric (IBP)
        """

        self.alpha = alpha
        self.beta = beta

        self.nonparametric = nonparametric

        self._initialize_Z(Z, likelihood)
        self._initialize_ranges()

        if nonparametric:
            self._poisson_param = float(alpha/self._num_of_objects)


    def _initialize_ranges(self):
        self._num_of_objects, self._num_of_latfeats = self._Z.shape

        self._obj_range = range(self._num_of_objects)
        self._latfeat_range = range(self._num_of_latfeats)

    def _initialize_Z(self, Z, likelihood):
        self._Z = np.where(Z > .5, 1, 0)

        likelihood.link_prior_params(Z_inference=self)
        self._log_likelihood = likelihood.compute_log_likelihood
        
        self._update_feature_counts()

        if self.nonparametric:
            self._compress_other_matrices = likelihood.compress_other_matrices
            self._append_to_other_matrices = likelihood.append_to_other_matrices

    def _update_feature_counts(self, z_new=None, obj=None, latfeat=None):
        if z_new == None:
            self._feature_counts = self._Z.sum(axis=0)
            self._update_log_prior_values()

        elif z_new != self._Z[obj, latfeat]:
            self._feature_counts[obj, latfeat] += self._Z[obj, latfeat] - z_new
            self._update_log_prior_values(obj)

    def _update_log_prior_values(self, obj=None):
        V = self._Z.shape[0]

        if self.nonparametric:
            smoothing = np.array([V-1., 0.])
        else:
            smoothing = np.array([self.beta + V - 1., self.alpha])

        if obj == None:
            feature_count_minus_obj = self._feature_counts[None,:] - self._Z
            counts_smoothed = smoothing[:,None,None] + np.array([-1, 1])[:,None,None] * feature_count_minus_obj

            self._log_prior_values = np.log(counts_smoothed)

        else:
            feature_count_minus_obj = self._feature_counts - self._Z[obj]
            counts_smoothed = smoothing[:,None] + np.array([-1, 1])[:,None] * feature_count_minus_obj

            self._log_prior_values[obj] = np.log(counts_smoothed)

    def _compute_log_posterior(self, z_new, obj, latfeat):
        log_likelihood = self._log_likelihood(z_new=z_new, obj=obj, latfeat=feat)

        return self._log_prior_values[z_new, obj, latfeat] + log_likelihood


    def _compress(self):
        latfeats_gt_0 = self.feature_counts > 0

        self._Z = self._Z.compress(latfeats_gt_0, axis=1)
        self.feature_counts = self.feature_counts.compress(latfeats_gt_0)

        self._compress_other_matrices(latfeats_gt_0)

    def _append_new_latfeats(self, obj):
        num_of_new_latfeats = sp.stats.poisson.rvs(mu=self._poisson_param)

        if num_of_new_latfeats:
            new_latfeats = np.zeros([self._num_of_objects, num_of_new_latfeats])
            new_latfeats[obj] += 1

            self._Z = np.append(self._Z, new_latfeats, axis=1)
            self.feature_counts = np.append(self.feature_counts, 
                                            np.ones(num_of_new_latfeats))

            self._latfeat_range = range(self._Z.shape[1])
            self._append_to_other_matrices(num_of_new_latfeats)

    def get_num_of_latfeats(self):
        return self._Z.shape[1]

    def get_param(self):
        return self._Z

    def _sample(self, obj, latfeat):
        logpost_on = self._compute_log_posterior(1, obj, latfeat)
        logpost_off = self._compute_log_posterior(0, obj, latfeat)

        prob = logpost_on / np.logaddexp(logpost_on, logpost_off)

        return sp.stats.bernoulli.rvs(prob)

    def sample(self):
        for obj in self._obj_range:
            self._update_prior_values(obj)

            for latfeat in self._latfeat_range:
                new = self._sample(obj, latfeat)
                self._update_feature_counts(new, obj, latfeat)
                self._Z[obj, latfeat] = new

            if self.nonparametric:
                self._compress()
                self._append_new_latfeats(obj)


class BSampler(Sampler):

    def __init__(self, B, likelihood, lmbda, proposal_bandwidth):
        self.lmbda = lmbda

        self._initialize_B(B, likelihood)
        self._initialize_ranges()
        self._initialize_proposer(proposal_bandwidth)

    def _initialize_B(self, B, likelihood):
        self._log_likelihood = likelihood.compute_log_likelihood
        self._B = B

        likelihood.link_prior_params(B_inference=self)

    def _initialize_ranges(self):
        self._num_of_latfeats, self._num_of_obsfeats = self._B.shape

        self._latfeat_range = range(self._num_of_latfeats)
        self._obsfeat_range = range(self._num_of_obsfeats)

    def _initialize_proposer(self, proposal_bandwidth):
        self._proposal_bandwidth = proposal_bandwidth
        self._proposal_window = self._B / proposal_bandwidth
        self._log_proposal_prob_values = np.log(self._B)

    def _compute_lp_b_new(self, b_new, latfeat, obsfeat):
        ll =  self._log_likelihood(b_new=b_new, 
                                   latfeat=latfeat, 
                                   obsfeat=obsfeat)
        lp = -self.lmbda * b_new

        return log_like + log_prior

    def _compute_lp_B(self, B):
        ll =  self._log_likelihood(B=B)
        lp = -self.lmbda * B

        return log_like + log_prior            

    def _compute_log_posterior(self, B=None, b_new=None, latfeat=None, obsfeat=None):
        if b_new != None:
            return self._compute_lp_b_new(b_new, obj, obsfeat)

        elif B != None:
            return self._compute_lp_B(B)

        else:
            return self._log_posterior_values[latfeat, obsfeat]

    def _proposer(self, latfeat, obsfeat):
        window_size = self._proposal_window[latfeat, obsfeat]
        
        return np.random.uniform(low=self._B[latfeat,obsfeat]-window_size, 
                                 high=self._B[latfeat,obsfeat]+window_size)

    def _log_proposal_prob(self, latfeat, obsfeat, proposal=None):
        if proposal == None:
            return self._log_proposal_prob_values[latfeat, obsfeat]

        else:
            return np.log(proposal)

    def _acceptance_prob(self, proposal, latfeat, obsfeat):
        current_log_post_val = self._compute_log_posterior(latfeat, obsfeat) 
        proposal_log_post_val = self._compute_log_posterior(latfeat, obsfeat, proposal) 

        log_post_ratio = current_log_post_val - proposal_log_post_val 

        ## transition probabilities are confusing:
        ## the first is really log(p(new->old))
        ## and the second is really log(p(old->new))
        ## without constants
        current_log_prop_val = self._log_proposal_prob(latfeat, obsfeat) 
        proposal_log_prop_val = self._log_proposal_prob(latfeat, obsfeat, proposal) 

        log_proposal_ratio = current_log_prop_val - proposal_log_prop_val

        return log_post_ratio + log_proposal_ratio

    def _update_B(proposal, latfeat, obsfeat):
        self._B[latfeat, obsfeat] = proposal

        self._proposal_window[latfeat, obsfeat] = proposal / self._proposal_bandwidth
        self._log_proposal_prob_values[latfeat, obsfeat] = np.log(proposal)

    def _sample(self, latfeat, obsfeat):
        proposal = self._proposer(latfeat, obsfeat)

        acceptance_log_prob = self._acceptance_prob(proposal, latfeat, obsfeat)
        acceptance_log_prob = np.min([0, acceptance_prob])

        accept = np.log(np.random.uniform(low=0., high=1.)) < acceptance_log_prob

        if accept:
            self._update_B(proposal, latfeat, obsfeat)

    def _update_log_posterior_values(self):
        log_like_vals = self._log_likelihood(B=self._B)
        log_prior_vals = -self.lmbda*self._B

        self._log_posterior_values = log_prior_vals + log_like_vals

    def sample(self):
        self._update_log_posterior_values()

        for latfeat in self._latfeat_range:
            for obsfeat in self._obsfeat_range:
                self._sample(latfeat, obsfeat)

    def get_param(self):
        return self._B

    def compress(self, latfeat_gt_0):
        self._B = self._B[latfeat_gt_0]



################
## optimizers ##
################

class Optimizer(object):
    
    @abc.abstractmethod
    def optimize(self):
        return

class JointOptimizer(Optimizer):

    def __init__(self, data, learning_rate=0.99, gamma=1., lmbda=1.):
        self._X = data.get_data().astype(theano.config.floatX)

        self._learning_rate = learning_rate
        self._gamma = gamma
        self._lmbda = lmbda

    def _initialize_shared_variables(self):
        obj, obsfeat = self._X.shape

        D = np.random.beta(1., 1., [obj, obsfeat]).astype(theano.config.floatX)
        Z = np.random.beta(1., 1., [obj, latfeat]).astype(theano.config.floatX)
        B = np.random.exponential(1., [latfeat, obsfeat]).astype(theano.config.floatX)

        self._tX = theano.shared(self._X, name='X')
        self._tD = theano.shared(D, name='D')
        self._tZ = theano.shared(Z, name='Z')
        self._tB = theano.shared(B, name='B')

        self._tZB = T.dot(tZ, tB)

    def optimize(self, X, latfeat, maxiter, subiter):
        gamma_poisson_numer = T.sum(self._tX * T.log(self._tD), axis=1)
        gamma_poisson_denom = (1 + T.sum(self._tX, axis=1))*T.log(self._gamma+T.sum(self._tD, axis=1))

        gamma_poisson = T.sum(gamma_poisson_numer - gamma_poisson_denom)
        beta_D = T.sum(T.log(self._tZB) + (self._tZB-1.)*T.log(self._tD))
        exp_B = T.sum(-self._lmbda*self._tB)

        log_likelihood = theano.function([], gamma_poisson)
        log_posterior = theano.function([], gamma_poisson+beta_D+exp_B)

        gradient_MLE_D = T.tanh(T.grad(gamma_poisson, self._tD))
        gradient_MAP_D = T.tanh(T.grad(beta_D + gamma_poisson, self._tD))
        gradient_Z = T.tanh(T.grad(beta_D, self._tZ))
        gradient_B = T.tanh(T.grad(exp_B + beta_D, self._tB))

        step_size_D = learning_rate * (T.minimum(self._tD, 1.-self._tD) - 1e-20)
        step_size_Z = learning_rate * (T.minimum(self._tZ, 1.-self._tZ) - 1e-20)
        step_size_B = learning_rate * T.minimum(self._tB, T.abs_(T.log(self._tB))+1.)

        update_MLE_D = theano.function(inputs=[],
                                       outputs=[],
                                       updates={tD:tD+step_size_D*gradient_MLE_D},
                                       name='update_MLE_D')
        update_MAP_D = theano.function(inputs=[],
                                       outputs=[],
                                       updates={tD:tD+step_size_D*gradient_MAP_D},
                                       name='update_MAP_D')
        update_Z = theano.function(inputs=[],
                                   outputs=[],
                                   updates={tZ:tZ+step_size_Z*gradient_Z},
                                   name='update_Z')
        update_B = theano.function(inputs=[],
                                   outputs=[],
                                   updates={tB:tB+step_size_B*gradient_B},
                                   name='update_B')

        for i in np.arange(subiter):
            # sys.stdout.write("unnormalized log likelihood: %d%%   \r" % (log_likelihood()) )
            # sys.stdout.flush()

            update_MLE_D()

        for i in np.arange(maxiter):
            for j in np.arange(subiter):
                update_B()
                update_Z()
            for j in np.arange(subiter):
                update_MAP_D()

        return self._tD.get_value(), self._tZ.get_value(), self._tB.get_value()

class DOptimizer(Optimizer):

    def __init_(self):
        raise NotImplementedError

    def _MAP_optimize(self):
        D_shape = self._D.shape

        lp = lambda D: self._log_likelihood(D) + self._log_prior(D)
        lp_flattened = lambda D: -lp(D.reshape(D_shape))

        lp_jac = lambda D: self._log_likelihood_jacobian(D) + self._log_prior_jacobian(D)
        lp_jac_flattened = lambda D: -lp_jac(D.reshape(D_shape)).flatten()

        bounds = [(10e-20,1.-10e-20)]*np.prod(D_shape)

        return minimize(fun=lp_flattened,
                        x0=initial_D.flatten(),
                        jac=lp_jac_flattened,
                        bounds=bounds,
                        method=self.optimizer).reshape(D_shape)

    def initialize_prior(self, prior):
        self._log_prior = prior.compute_log_prior
        self._log_prior_jacobian = prior.compute_log_prior_jacobian



class BOptimizer(Optimizer):

    def __init_(self):
        raise NotImplementedError


####################
## fitting procedure
####################

class Model(object):

    @abc.abstractmethod
    def fit(self, iterations, burnin, thinning):
        return


class FullGibbs(Model):

    def __init__(self, data, num_of_latfeats):
        
        num_of_latfeats = num_of_latfeats if num_of_latfeats else 100

        initial_D, initial_Z, initial_B = initialize_D_Z_B(data.get_data(),
                                                           num_of_latfeats,
                                                           maxiter=100,
                                                           subiter=100)

        data_likelihood = PoissonGammaProductLikelihood(data=data,
                                                        gamma=args.gamma)

        print 'initializing D sampler'

        D_sampler = DSampler(D=initial_D,
                             likelihood=data_likelihood, 
                             proposal_bandwidth=args.distributionproposalbandwidth)

        beta_posterior = BetaPosterior(D_inference=D_sampler)

        print 'initializing Z sampler'

        Z_sampler = ZSampler(Z=initial_Z,
                             likelihood=beta_posterior,
                             alpha=args.alpha,
                             nonparametric=args.nonparametric)

        print 'initializing B sampler'

        B_sampler = BSampler(B=initial_B,
                             likelihood=beta_posterior,
                             lmbda=args.lmbda,
                             proposal_bandwidth=args.featloadingsproposalbandwidth)


        self._D_sampler = D_sampler
        self._Z_sampler = Z_sampler
        self._B_sampler = B_sampler

    def _initialize_samples(self, iterations, burnin, thinning):
        self._D_samples = np.empty((iterations-burnin)/thinning, 
                                   dtype=object)
        self._Z_samples = np.empty((iterations-burnin)/thinning, 
                                   dtype=object)
        self._B_samples = np.empty((iterations-burnin)/thinning, 
                                   dtype=object)

    def _sample(self):
        self._D_sampler.sample()
        self._Z_sampler.sample()
        self._B_sampler.sample()

    def _save_samples(self, iterations, burnin, thinning):
        self._D_samples[(itr-burnin)/thinning] = self._D_sampler.get_param()
        self._Z_samples[(itr-burnin)/thinning] = self._Z_sampler.get_param()
        self._B_samples[(itr-burnin)/thinning] = self._B_sampler.get_param()

    def fit(self, iterations, burnin, thinning):
        self._initialize_samples(iterations, burnin, thinning)

        for itr in np.arange(iterations):
            self._sample()

            if itr >= burnin and not itr % thinning:
                self._save_samples(iterations, burnin, thinning)
