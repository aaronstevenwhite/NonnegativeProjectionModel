import abc, warnings
import numpy as np
import scipy as sp
import theano

from scipy.optimize import minimize
from sklearn.decomposition import ProjectedGradientNMF

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

    def __init__(self, data_fname, gamma=1.):
        ## gamma distribution parameters
        self.gamma = gamma

        self._load_X(data_fname)

    def _load_X(self, fname):
        data = np.loadtxt(fname, 
                          dtype=str, 
                          delimiter=',')

        self.objects = data[1:,0]
        self.features = data[0,1:]
        self._X = data[1:,1:].astype(int)

        self._X_obj_count = self._X.sum(axis=1)

        self._obj_range = range(self._X.shape[0])
        self._obsfeat_range = range(self._X.shape[1])

    def get_data(self):
        return self._X

    # def get_X_count(self, obj, obsfeat=None):
    #     if obsfeat == None:
    #         return self._X_obj_count[obj]
    #     else:
    #         return self._X[obj, obsfeat]

    def compute_log_likelihood(self, D, obj=None, obsfeat=None):
        obj = self._obj_range if obj == None else obj
        obsfeat = self._obsfeat_range if obsfeat == None else obsfeat

        log_numer = np.sum(self._X[obj][:,obsfeat] * np.log(D[obj][:,obsfeat]), axis=1)

        log_denom_X = 1 + self._X_obj_count[obj]
        log_denom_D = np.log(self.gamma + D[obj].sum(axis=1))
        log_denom = log_denom_X * log_denom_D

        return np.sum(log_numer - log_denom)

    def compute_log_likelihood_jacobian(self, D, obj=None, obsfeat=None):
        obj = self._obj_range if obj == None else obj
        obsfeat = self._obsfeat_range if obsfeat == None else obsfeat

        first_term = self._X[obj][:,obsfeat] / D[obj][:,obsfeat]

        second_term_numer = (1 + self._X_obj_count[obj])[:,None]
        second_term_denom = (self.gamma + D[obj].sum(axis=1))[:,None]
        second_term = second_term_numer / second_term_denom

        return first_term - second_term


class BetaPosterior(Prior, Likelihood):

    def __init__(self, D_inference):
        self._D = D_inference.get_param()
        D_inference.link_prior(self)

        self._Z = None
        self._B = None

    def link_prior_params(self, Z_inference=None, B_inference=None):
        self._Z_inference = Z_inference
        self._B_inference = B_inference

        self._Z = Z_inference.get_param() if Z_inference != None else None
        self._B = B_inference.get_param() if B_inference != None else None

        if self._Z != None and self._B != None:
            self._ZB = np.dot(self._Z, self._B)
            self._update_log_likelihood_values()

    def compress_other_matrices(self, latfeats_gt_0):
        self._B_inference.compress(latfeats_gt_0)

    def append_to_other_matrices(self, num_of_new_features):
        self._B_inference.sample_new_features(num_of_new_features)


    def get_data(self):
        return self._D

    def _compute_ll_z_new(self, z_new, obj, latfeat):
        z_obj = np.insert(np.delete(self._Z[obj,:], 
                                    latfeat), 
                          latfeat, 
                          z_new)

        ZB_obj = np.dot(z_obj, self._B)

        ll = np.log(ZB_obj) + (ZB_obj[latfeat] - 1)*np.log(self._D[obj])
            
        return np.sum(ll)

    def _compute_ll_b_new(self, b_new, latfeat, obsfeat):
        b_obsfeat = np.insert(np.delete(self._B[:,obsfeat], 
                                        latfeat), 
                              latfeat, 
                              b_new)

        ZB_obsfeat = np.dot(self._Z, b_obsfeat)

        ll = np.log(ZB_obsfeat) + (ZB_obsfeat[latfeat] - 1)*np.log(self._D[:,obsfeat])

        return np.sum(ll)

    def _compute_ll_B(self, B):
        ZB = np.dot(self._Z, B)
        self._log_likelihood = np.log(ZB) + (ZB - 1) * np.log(self._D)

        return np.sum(self._log_likelihood)

    def compute_log_likelihood(B=None, z_new=None, b_new=None, 
                               obj=None, latfeat=None, obsfeat=None):
        if z_new != None:
            return self._compute_ll_z_new(z_new, obj, latfeat)

        elif b_new != None:
            return self._compute_ll_b_new(b_new, latfeat, obsfeat)

        else:
            ## this would not be necessary if we weren't
            ## doing optimization for B since it could be
            ## done using self._B
            return self._compute_ll_B(B)

    def _update_log_likelihood_values(self):
        self._log_likelihood = np.log(self._ZB) + (self._ZB - 1) * np.log(self._D)

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
    
    def __init__(self, likelihood, proposal_bandwidth=1., optimizer='L-BFGS-B'):
        self._log_likelihood = likelihood.compute_log_likelihood

        ## this smoother should really be data dependent 
        ## so as not to yield something below the optimizer bounds
        X = likelihood.get_data() + 10e-10 
        initial_D = X.astype(float)/X.sum(axis=1)[:,None]

        jacobian = likelihood.compute_log_likelihood_jacobian
        self._D = self._MLE_optimize(initial_D, jacobian, optimizer)

        ## initialize ranges so they don't have to 
        ## be recomputed each time a range is needed
        self._obj_range = range(self._D.shape[0])
        self._obsfeat_range = range(self._D.shape[1])

        ## find point-wise min of D and 1-D
        D_minimum = np.minimum(self._D, 1-self._D)

        ## use min to precompute proposal window and transition probs
        self._proposal_bandwidth = proposal_bandwidth
        self._proposal_window = D_minimum / proposal_bandwidth
        self._log_proposal_prob_values = np.log(D_minimum)

    def _MLE_optimize(self, initial_D, jacobian, optimizer):
        if optimizer:
            D_shape = initial_D.shape

            ll_flattened = lambda D: -self._log_likelihood(D.reshape(D_shape))
            ll_jac_flattened = lambda D: -jacobian(D.reshape(D_shape)).flatten()

            bounds = [(1e-20,1.-1e-20)]*np.prod(D_shape)

            solution = minimize(fun=ll_flattened,
                                x0=initial_D.flatten(),
                                jac=ll_jac_flattened,
                                bounds=bounds,
                                method=optimizer)

            return solution.x.reshape(D_shape)
        
        else:
            warnings.warn('No optimizer specified; D will not be initialized with optimization')
            return initial_D

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

        return log_like + log_prior

    def _compute_lp_D(self, D):
        ll =  self._log_likelihood(D=D)
        lp = self._log_prior(D=D)            

        return log_like + log_prior            

    def _compute_log_posterior(self, D=None, d_new=None, obj=None, obsfeat=None):
        if d_new != None:
            return self._compute_lp_d_new(d_new, obj, obsfeat)

        elif D != None:
            return self._compute_lp_D(D)

        else:
            return self._log_posterior_values[obj, obsfeat]

    def _proposer(self, obj, obsfeat):
        window_size = self._proposal_window[obj, obsfeat]
        
        return np.random.uniform(low=d_mn_old-window_size, 
                                 high=d_mn_old+window_size)

    def _log_proposal_prob(self, obj, obsfeat, proposal=None):
        if proposal == None:
            return self._log_proposal_prob_values[obj, obsfeat]

        else:
            return np.log(np.min([proposal, 1-proposal]))

    def _acceptance_prob(self, proposal, obj, obsfeat):
        current_log_post_val = self._compute_log_posterior(obj, obsfeat) 
        proposal_log_post_val = self._compute_log_posterior(obj, obsfeat, proposal) 

        log_post_ratio = current_log_post_val - proposal_log_post_val 

        ## transition probabilities are confusing:
        ## the first is really log(p(new->old))
        ## and the second is really log(p(old-new))
        ## without constants
        current_log_prop_val = self._log_proposal_prob(obj, obsfeat) 
        proposal_log_prop_val = self._log_proposal_prob(obj, obsfeat, proposal) 

        log_proposal_ratio = current_log_prop_val - proposal_log_prop_val

        return log_post_ratio + log_proposal_ratio

    def _sample(obj, obsfeat):
        proposal = self._proposer(obj, obsfeat)

        acceptance_log_prob = self._acceptance_prob(proposal, obj, obsfeat)
        acceptance_log_prob = np.min([0, acceptance_prob])

        accept = np.log(np.random.uniform(low=0., high=1.)) < acceptance_log_prob

        if accept:
            self._update_D(proposal, obj, obsfeat)

    def _update_D(proposal, obj, obsfeat):
        self._D[obj, obsfeat] = proposal

        d_obj_obsfeat_min = np.min([proposal, 1-proposal])
        self._proposal_window[obj, obsfeat] = d_obj_obsfeat_min / self.proposal_bandwidth
        self._log_proposal_prob_values[obj, obsfeat] = np.log(d_obj_obsfeat_min)

    def _update_log_posterior_values(self):
        log_prior_vals = self._log_prior(self._obj_range, self._obsfeat_range)
        log_post_vals = self._log_likelihood(self._D)

        self._log_posterior_values = log_prior_vals + log_post_vals

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
    
    def __init__(self, likelihood, alpha, beta=None, num_of_latfeats=0, optimize=True):
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

        self._num_of_objects = likelihood.get_data().shape[0]
        self._poisson_param = float(alpha/self._num_of_objects)

        self._initialize_Z(likelihood, num_of_latfeats, optimize)
        self._initialize_likelihood(likelihood)

    def _initialize_Z(self, likelihood, num_of_latfeats, optimize):
        D = likelihood.get_data()

        if num_of_latfeats:
            self._num_of_latfeats = num_of_latfeats
        else:
            self._num_of_latfeats = D.shape[1]/10

        if optimize:
            self._Z = self._thresholded_nmf(D)
        else:
            feature_prob = 1. / (1+np.exp(-self.alpha))
            self._Z = sp.stats.bernoulli.rvs(feature_prob, 
                                             size=[self._num_of_objects, 
                                                   self._num_of_latfeats])

        self._set_feature_counts()
        self._set_log_counts_minus_obj()

    def _thresholded_nmf(self, D):
        nmf_model = ProjectedGradientNMF(n_components=self._num_of_latfeats, 
                                         init='random')
        nmf_score_matrix = nmf_model.fit_transform(-1./np.log(D))

        threshold = np.percentile(nmf_score_matrix, 
                                  100. / (1+np.exp(-self.alpha)))
        
        return np.where(nmf_score_matrix > threshold, 1., 0.)

    def _initialize_likelihood(self, likelihood):
        likelihood.link_prior_params(Z_inference=self)
        
        self._log_likelihood = likelihood.compute_log_likelihood
        self._compress_other_matrices = likelihood.compress_other_matrices
        self._append_to_other_matrices = likelihood.append_to_other_matrices


    def _set_feature_counts(self):
        self._feature_counts = self._Z.sum(axis=0)

    def _update_feature_counts(self, new, obj, feat):
        if new != self._Z[obj, latfeat]:
            self._feature_counts[obj, feat] += self._Z[obj, feat] - new

    def _set_log_counts_minus_obj(self):
        self._log_counts_minus_obj = np.log(self._feature_counts[None,:] - self._Z)


    def _update_log_counts_minus_obj(self, obj):
        self._log_counts_minus_obj[obj] = np.log(self._feature_counts - self._Z[obj])


    def _compute_log_posterior(self, val, obj, latfeat):
        log_likelihood = self._log_likelihood(val=val, obj=obj, latfeat=feat)

        return self._log_counts_minus_obj[obj, latfeat] + log_likelihood

    def _sample_val(self, obj, latfeat):
        logpost_on = self._compute_log_posterior(1, obj, latfeat)
        logpost_off = self._compute_log_posterior(0, obj, latfeat)

        prob = logpost_on / np.logaddexp(logpost_on, logpost_off)

        return sp.stats.bernoulli.rvs(prob)

    def _compress_Z(self):
        latfeats_gt_0 = self.feature_counts > 0

        self._Z = self._Z.compress(latfeats_gt_0, axis=1)
        self.feature_counts = self.feature_counts.compress(latfeats_gt_0)

        self._compress_other_matrices(latfeats_gt_0)

    def _append_new_latent_features(self, obj):
        num_of_new_latent_features = sp.stats.poisson.rvs(mu=self._poisson_param)

        if num_of_new_latent_features:
            new_latent_features = np.zeros([num_of_objects, num_of_new_latent_features])
            new_latent_features[obj] += 1

            self._Z = np.append(self._Z, new_latent_features, axis=1)
            self.feature_counts = np.append(self.feature_counts, 
                                            np.ones(num_of_new_latent_features))

            self._append_to_other_matrices(num_of_new_latent_features)

    def _reorder_Z(self):
        raise NotImplementedError
        self._reorder_other_matrices(reordered_indices)

    def _update_Z_obj(self, obj):
        self._update_log_counts_minus_obj(obj)

        for latfeat in np.arange(self.num_of_latent_features):
            new = self._sample_val(obj, latfeat)
            self._update_feature_counts(new, obj, latfeat)
            self._Z[obj, latfeat] = new

            self._compress_Z()
            self._append_new_latent_features(obj)
            #self._reorder_Z()


    def _update_Z(self):

        for obj in np.arange(self.num_of_objects):
            self._update_Z_obj(obj)

    def get_num_of_latfeats(self):
        return self._Z.shape[1]

class BSampler(Sampler):

    def __init_(self, likelihood, lam, num_of_latfeats):
        ## initialize ranges so they don't have to 
        ## be recomputed each time a range is needed
        self._latfeat_range = range(num_of_latfeats)
        self._obsfeat_range = range(self._D.shape[1])

        raise NotImplementedError

    def _initialize_likelihood(self, likelihood):
        likelihood.link_prior_params(B_inference=self)
        
        self._log_likelihood = likelihood.compute_log_likelihood

    def compress(self, latfeat_gt_0):
        self._B = self._B[latfeat_gt_0]

    def sample(self):
        self._update_log_posterior_values()

        for latfeat in self._latfeat_range:
            for obsfeat in self._obsfeat_range:
                self._sample(latfeat, obsfeat)



################
## optimizers ##
################

class Optimizer(object):
    
    @abc.abstractmethod
    def _MLE_optimize(self):
        return

    @abc.abstractmethod
    def _MAP_optimize(self):
        return

    @abc.abstractmethod
    def optimize(self):
        return

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

    def __init__(self, D_sampler, Z_Sampler, B_Sampler):
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


class BSampler(Sampler):

    def __init_(self, likelihood, lam, num_of_latfeats):
        ## initialize ranges so they don't have to 
        ## be recomputed each time a range is needed
        self._latfeat_range = range(self._D.shape[0])
        self._obsfeat_range = range(self._D.shape[1])

        raise NotImplementedError

    def _initialize_likelihood(self, likelihood):
        likelihood.link_prior_params(B_inference=self)
        
        self._log_likelihood = likelihood.compute_log_likelihood_B

    def compress(self, latfeat_gt_0):
        self._B = self._B[latfeat_gt_0]

    def sample_new_features(self, num_of_new_features):
        raise NotImplementedError

    def sample(self):
        self._update_log_posterior_values()

        for obj in self._latfeat_range:
            for obsfeat in self._obsfeat_range:
                self._sample(obj, obsfeat)
