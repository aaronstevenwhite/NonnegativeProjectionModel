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
        self._B_inference.compress(self._Z, latfeats_gt_0)

    def append_to_other_matrices(self, num_of_new_features):
        self._B_inference.append_new_features(num_of_new_features)


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

    def append_new_features(self, num_of_new_features):
        shape = [num_of_new_features, self._num_of_obsfeats]
        new_loadings = np.random.exponential(1., shape).astype(theano.config.floatX)
        self._B = self._B.append(new_loadings, axis=0)



################
## optimizers ##
################

class Optimizer(object):

    def __init__(self, data, learning_rate=0.99):
        self._data = data.get_data().astype(theano.config.floatX)
        self._learning_rate = learning_rate
        self._num_of_rows, self._num_of_columns = self._X.shape
    
    @abc.abstractmethod
    def optimize(self):
        return

    @abc.abstractmethod
    def _construct_shared(self):
        return 


class DOptimizer(Optimizer):

    def __init__(self, X, gamma):
        self._X = X.astype(theano.config.floatX)
        self._D = np.random.beta(1., 1., X.shape).astype(theano.config.floatX)
        
        self._tX = theano.shared(self._X, name='X')
        self._tD = theano.shared(self._D, name='D')

        self._likelihood_objective = self._construct_likelihood_objective(gamma)
        self._MLE_updater = self._construct_updater(self._likelihood_objective)

    def get_tD(self):
        return self._tD

    def _construct_likelihood_objective(self, gamma):
        tX, tD = self._tX, self._tD

        gamma_poisson_numer = T.sum(tX * T.log(tD), axis=1)
        gamma_poisson_denom = (1 + T.sum(tX, axis=1))*T.log(gamma+T.sum(tD, axis=1))

        return T.sum(gamma_poisson_numer - gamma_poisson_denom)

    def _construct_prior_objective(self, tZB):
        tD = self._tD

        return T.sum(T.log(tZB) + (tZB-1.)*T.log(tD))

    def _construct_posterior_objective(self, tZB):
        tX, tD = self._tX, self._tD

        log_prior = self._construct_prior_objective(tZB)
        log_likelihood = self._likelihood_objective

        return log_prior + log_likelihood

    def _construct_updater(self, objective):
        tD = self._tD

        gradient = T.tanh(T.grad(objective, tD))

        step_size_D = learning_rate * (T.minimum(tD, 1.-tD)-1e-20)

        return theano.function(inputs=[],
                               outputs=[],
                               updates={tD:tD+step_size_D*gradient},
                               name='update_D')

    def set_prior_param(self, tZB):
        self._tZB = tZB
        self._posterior_objective = self._construct_posterior_objective(tZB)
        self._MAP_updater = self._construct_updater(self._posterior_objective)

    def get_prior_objective(self):
        return self._construct_prior_objective

    def get_param(self):
        return self._D
    
    def optimize(self, maxiter=100, objective='MAP'):
        if objective_type=='MAP':
            updater = self._MAP_updater  
        else:
            updater = self._MLE_updater
            
        for i in np.arange(maxiter):
            updater()

        self._D = self._tD.get_value()
            
class ZBOptimizer(Optimizer):

    def __init__(self, D_optimizer, num_of_objects, num_of_obsfeats, num_of_latfeats, lmbda):
        shape_Z = [num_of_objects, num_of_latfeats]
        shape_B = [num_of_latfeats, num_of_obsfeats]

        self._Z = np.random.beta(1., 1., shape_Z).astype(theano.config.floatX)
        self._B = np.random.exponential(1., shape_B).astype(theano.config.floatX)

        self._tZ = theano.shared(self._Z, name='Z')
        self._tB = theano.shared(self._B, name='B')
        self._tZB = T.dot(self._tZ, self._tB)

        D_optimizer.set_prior_param(self._tZB)
        
        self._posterior_objective = self._construct_posterior_objective(D_optimizer, lmbda)
        self._MAP_updater_Z, self._MAP_updater_B = self._construct_updater(self._posterior_objective)
        
    def _construct_prior_objective(self, lmbda):
        return T.sum(-lmbda*self._tB)
        
    def _construct_likelihood_objective(self, D_optimizer):
        return D_optimizer.get_prior_objective()

    def _construct_posterior_objective(self, D_optimizer, lmbda):

        log_prior = self._construct_prior_objective(lmbda)
        log_likelihood = self._construct_likelihood_objective(D_optimizer)

        return log_prior + log_likelihood

    def _construct_updater(self, objective):
        tZ = self._tZ
        tB = self._tB

        gradient_Z = T.tanh(T.grad(objective, tZ))
        gradient_B = T.tanh(T.grad(objective, tB))

        step_size_Z = learning_rate * (T.minimum(self._tZ, 1.-self._tZ) - 1e-20)
        step_size_B = learning_rate * T.minimum(tB, T.abs_(T.log(tB))+1.)

        update_Z = theano.function(inputs=[],
                                   outputs=[],
                                   updates={tZ:tZ+step_size_Z*gradient_Z},
                                   name='update_Z')
        update_B = theano.function(inputs=[],
                                   outputs=[],
                                   updates={tB:tB+step_size_B*gradient},
                                   name='update_B')

        return update_Z, update_B

    def _reset_Z(self, Z):
        self._Z = Z
        self._tZ.set_value(self._Z)

    def compress(self, Z, latfeats_gt_0):
        self._reset_Z(Z)
        self._B = self._B[latfeats_gt_0]

    def append_new_features(self, num_of_new_features):
        shape = [num_of_new_features, self._B.shape[1]]
        new_loadings = np.random.exponential(1., shape).astype(theano.config.floatX)
        self._B = self._B.append(new_loadings, axis=0)

        self._tB.set_value(self._B)
        
    def get_param(self):        
        return self._Z, self._B
    
    def optimize(self, maxiter=100, subiter=100, update_Z=False):
        updater_Z, updater_B = self._MAP_updater_Z, self._MAP_updater_B  

        for i in np.arange(maxiter):
            if update_Z:
                for j in np.arange(subiter):
                    updater_Z()
            for j in np.arange(subiter):
                updater_B()

        self._Z, self._B = self._tZ.get_value(), self._tB.get_value()

class JointOptimizer(Optimizer):

    def __init__(self, data, num_of_latfeats, gamma, lmbda):
        X = data.get_data()
        num_of_objects, num_of_obsfeats = X.shape

        self._D_optimizer = DOptimizer(X, gamma)
        self._ZB_optimizer = ZBOptimizer(D_optimizer,
                                         num_of_objects,
                                         num_of_latfeats,
                                         num_of_obsfeats,
                                         lmbda)

    def get_param(self):
        D = self._D_optimizer.get_param()
        Z, B = self._ZB_optimizer.get_param()

        return D, Z, B
    
    def get_optimizers(self):
        return self._D_optimizer, self._ZB_optimizer
    
    def optimize(self, maxiter, subiter):
        ## initialize D with MLE estimate
        self._D_optimizer.optimize(maxiter=subiter, objective='MLE')

        for i in np.arange(maxiter):
            self._ZB_optimizer(maxiter=1, subiter=subiter, update_Z=True)
            self._D_optimizer.optimize(maxiter=subiter, objective='MAP')



#######################
## fitting procedure ##
#######################

class Model(object):

    def __init__(self, data, num_of_latfeats, gamma, lmbda, sample_D, sample_B,
                 distributionproposalbandwidth, featloadingsproposalbandwidth,
                 maxiter, subiter, alpha, beta, lmbda, gamma, nonparametric):
        num_of_latfeats = num_of_latfeats if num_of_latfeats else 100
        joint_optimizer = JointOptimizer(data=data,
                                         num_of_latfeats=num_of_latfeats,
                                         gamma=gamma,
                                         lmbda=lmbda)

        joint_optimizer.optimize(maxiter=100, subiter=100)
        
        self._initialize_D_inference(data, joint_optimizer, sample_D, gamma,
                                     maxiter, subiter, distributionproposalbandwidth)

        beta_posterior = BetaPosterior(D_inference=self._D_inference)

        self._initialize_Z_inference(beta_posterior, joint_optimizer, alpha, beta, nonparametric)
        self._initialize_B_inference(beta_posterior, joint_optimizer, sample_B, lmbda,
                                     maxiter, subiter, featloadingsproposalbandwidth)
        
    def _initialize_D_inference(self, data, joint_optimizer, sample_D, gamma,
                                maxiter, subiter, proposal_bandwidth):

        initial_D, _, _ = joint_optimizer.get_param()
        D_optimizer, _ = joint_optimizer.get_optimizers()

        if sample_D:
            data_likelihood = PoissonGammaProductLikelihood(data=data,
                                                            gamma=gamma)

            self._D_inference = DSampler(D=initial_D,
                                         likelihood=data_likelihood, 
                                         proposal_bandwidth=proposal_bandwidth)

            self._fit_D = self._D_inference.sample
        else:
            self._D_inference = D_optimizer
            self._fit_D = lambda: self._D_inference.optimize(maxiter=maxiter, subiter=subiter)


    def _initialize_Z_inference(self, beta_posterior, initial_Z, alpha, beta, nonparametric):
        _, initial_Z, _ = joint_optimizer.get_param()
        
        self._Z_inference = ZSampler(Z=initial_Z,
                                     likelihood=likelihood,
                                     alpha=alpha,
                                     beta=beta,
                                     nonparametric=nonparametric)

        self._fit_Z = self._Z_inference.sample

    def _initialize_B_inference(self, likelihood, joint_optimizer, sample_B, lmbda, proposal_bandwidth):
        _, _, initial_B = joint_optimizer.get_param()
        _, ZB_optimizer = joint_optimizer.get_optimizers()
        
        if sample_B:
            self._B_inference = BSampler(B=initial_B,
                                         likelihood=likelihood,
                                         lmbda=lmbda,
                                         proposal_bandwidth=proposal_bandwidth)
            self._fit_B = self._B_inference.sample

        else:
            self._B_inference = ZB_optimizer
            self._fit_B = lambda: self._B_inference.optimize(maxiter=maxiter, subiter=subiter)

    
    def _initialize_samples(self, iterations, burnin, thinning):
        self._D_samples = np.empty((iterations-burnin)/thinning, 
                                   dtype=object)
        self._Z_samples = np.empty((iterations-burnin)/thinning, 
                                   dtype=object)
        self._B_samples = np.empty((iterations-burnin)/thinning, 
                                   dtype=object)

    def _fit(self):
        self._fit_D()
        self._fit_Z()
        self._fit_B()
        
    def _save_samples(self, iterations, burnin, thinning):
        self._D_samples[(itr-burnin)/thinning] = self._D_inference.get_param()
        self._Z_samples[(itr-burnin)/thinning] = self._Z_inference.get_param()
        self._B_samples[(itr-burnin)/thinning] = self._B_inference.get_param()

    def fit(self, iterations, burnin, thinning):
        self._initialize_samples(iterations, burnin, thinning)

        for itr in np.arange(iterations):
            self._fit()
            
            if itr >= burnin and not itr % thinning:
                self._save_samples(iterations, burnin, thinning)
