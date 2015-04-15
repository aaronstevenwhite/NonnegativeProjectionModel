import abc
import numpy as np
import scipy as sp
import theano

from scipy.optimize import minimize

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

class PoissonProductGammaLikelihood(Likelihood):

    def __init__(self, data_fname, gamma=1., delta=1.):
        ## gamma distribution parameters
        self.gamma = gamma
        self.delta = delta

        self._load_X(data_fname)

    def _load_X(self, fname):
        data = np.loadtxt(fname, 
                          dtype=str, 
                          delimiter=',')

        self.objects = data[1:,0]
        self.features = data[0,1:]
        self.X = data[1:,1:].astype(int)

        self._X_obj_count = self.X.sum(axis=1)

    def get_data(self):
        return X

    # def get_X_count(self, obj, obsfeat=None):
    #     if obsfeat == None:
    #         return self._X_obj_count[obj]
    #     else:
    #         return self._X[obj, obsfeat]

    def compute_log_likelihood(self, D, obj=None, obsfeat=None):
        obj = range(self.X.shape[0]) if obj == None else obj
        obsfeat = range(self.X.shape[1]) if obsfeat == None else obsfeat

        log_numer = np.sum(self.X[obj, obsfeat] * np.log(D[obj, obsfeat]), axis=1)

        log_denom_X = self.gamma + self._X_obj_count[obj]
        log_denom_D = np.log(self.delta + D[obj].sum(axis=1))
        log_denom = log_denom_X * log_denom_D

        return np.sum(log_numer - log_denom)

    def compute_log_likelihood_jacobian(self, D, obj=None, obsfeat=None):
        obj = range(self.X.shape[0]) if obj == None else obj
        obsfeat = range(self.X.shape[1]) if obsfeat == None else obsfeat

        first_term = self.X[obj,obsfeat] / D[obj,obsfeat]

        second_term_numer = (self.gamma + self._X_obj_count[obj])[:,None]
        second_term_denom = (self.delta + D[obj].sum(axis=1))[:,None]
        second_term = second_term_numer / second_term_denom

        return first_term - second_term


class BetaLikelihood(Likelihood):

    def __init__(self, D_sampler, Z_sampler, B_sampler):
        self._D = D_sampler.get_param()
        D_sampler.initialize_prior(self)

        self._Z = Z_sampler.get_param()
        self._B = B_sampler.get_param()

        self._ZB = np.dot(self._Z, self._B)

        self.compute_log_likelihood()

    def compute_log_likelihood(B=None, z_new=None, b_new=None, 
                               obj=None, latfeat=None, obsfeat=None):
        if z_new != None:
            z_obj = np.insert(np.delete(self._Z[obj,:], 
                                        latfeat), 
                              latfeat, 
                              z_new)

            ZB_obj = np.dot(z_obj, self._B)
            
            ll = np.log(ZB_obj) + (ZB_obj[latfeat] - 1)*np.log(self._D[obj])
            
            return np.sum(ll)

        elif b_new != None:
            b_obsfeat = np.insert(np.delete(self._B[:,obsfeat], 
                                            latfeat), 
                                  latfeat, 
                                  b_new)

            ZB_obsfeat = np.dot(self._Z, b_obsfeat)
        
            ll = np.log(ZB_obsfeat) + (ZB_obsfeat[latfeat] - 1)*np.log(self._D[:,obsfeat])

            return np.sum(ll)

        elif B != None:
            ZB = np.dot(self._Z, B)
            self._log_likelihood = np.log(ZB) + (ZB - 1) * np.log(self._D)
            
            return np.sum(self._log_likelihood)

        else:
            self._log_likelihood = np.log(self._ZB) + (self._ZB - 1) * np.log(self._D)

            return np.sum(self._log_likelihood)


    def compute_log_prior(self, D=None, d_new=None, obj=None, obsfeat=None):
        if d_new != None:
            try:
                assert None not in [obj, obsfeat]
            except:
                raise ValueError('if d_new specified, must specify obj and obsfeat')

            return (self._ZB[obj, obsfeat] - 1) * np.log(d_new)

        elif D != None:
            self._log_prior = (self._ZB - 1) * np.log(D)

            return np.sum(self._log_prior)

###########
## samplers
###########

class Sampler(object):
    
    @abc.abstractmethod
    def __init__(self, likelihood, **kwargs):
        return

    @abc.abstractmethod
    def get_param(self):
        return 

class DSampler(Sampler):
    
    def __init_(self, likelihood, proposal_bandwidth=1., optimizer='L-BFGS-B'):
        self._proposal_bandwidth = proposal_bandwidth
        self._initialize_D(likelihood, optimizer)

    def _initialize_D(self, likelihood, optimizer):
        self.optimizer = optimizer
        self._log_likelihood = likelihood.compute_log_likelihood_D

        X = likelihood.get_data()
        initial_D = X/X.sum(axis=1)[:,None]

        if optimizer:
            self._D = self._MLE_optimize(likelihood.compute_log_likelihood_jacobian_D)

        else:
            self._D = initial_D

        self._obj_range = range(self._D.shape[0])
        self._obsfeat_range = range(self._D.shape[1])

        D_minimum = np.minimum(self._D, 1-self._D)

        self._proposal_window = D_minimum / self._proposal_bandwidth
        self._log_proposal_prob_values = np.log(D_minimum)

    def _MLE_optimize(self, jacobian):
        self._log_likelihood_jacobian = jacobian

        D_shape = self._D.shape

        ll_flattened = lambda D: -self._log_likelihood(D.reshape(D_shape))
        ll_jac_flattened = lambda D: -jacobian(D.reshape(D_shape)).flatten()

        bounds = [(10e-20,1.-10e-20)]*np.prod(D_shape)

        return minimize(fun=ll_flattened,
                        x0=initial_D.flatten(),
                        jac=ll_jac_flattened,
                        bounds=bounds,
                        method=self.optimizer).reshape(D_shape)

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

    def _compute_log_posterior(self, obj, obsfeat, proposal=None):
        if proposal == None:
            return self._log_posterior_values[obj, obsfeat]

        else:
            log_like =  self._log_likelihood(d_new=proposal, 
                                             obj=obj, 
                                             obsfeat=obsfeat)
            log_prior = self._log_prior(d_new=proposal, 
                                        obj=obj, 
                                        obsfeat=obsfeat)

            return log_like + log_prior

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
        log_prior_vals = self._log_prior(self._D)
        log_post_vals = self._log_likelihood(self._D)

        self._log_posterior_values = log_prior_vals + log_post_vals

    def sample(self):
        self._update_log_posterior_values()

        for obj in self._obj_range:
            for obsfeat in self._obsfeat_range:
                self._sample(obj, obsfeat)

    def optimize(self):
        raise NotImplementedError

class BSampler(Sampler):

    def __init_(self):
        raise NotImplementedError


class ZSampler(Sampler):

    """
    Samples verb-by-latent binary features using Gibbs sampling

    Attributes:
      sentence (numpy.array(dtype=str)): the parsed sentence
    """
    
    def __init__(self, likelihood, alpha, num_of_latfeats=0):
        """
        Initialize the verb-by-latent binary feature sampler

        Args:
          likelihood (Likelihood): computes p(D|B,Z) and implements interfaces in ABC Likelihood
          alpha (float): positive real number parameterizing Beta(alpha, 1) or IBP(alpha)
          num_of_latfeats (int): number of latent binary features; 
                                 if 0, the sampler is nonparametric (IBP)
        """


        self.alpha = alpha
        self._num_of_objects = likelihood.get_num_of_objects()
        self._poisson_param = float(alpha/num_of_objects)

        self._initialize_Z(num_of_latfeats)
        self._initialize_likelihood(likelihood)


    def _initialize_Z(self, num_of_latfeats):
        if num_of_latfeats:
            self._num_of_latfeats = num_of_latfeats

        else:
            self._num_of_latfeats = 1 + sp.stats.poisson.rvs(mu=float(self.alpha))

        self.Z = sp.stats.bernoulli.rvs(.5, size=[self._num_of_objects, 
                                                  self._num_of_latfeats])

        self._set_feature_counts()
        self._set_log_counts_minus_obj()


    def _initialize_likelihood(self, likelihood):
        likelihood.set_Z(self.Z)
        
        self._log_likelihood = likelihood.compute_log_likelihood_Z
        self._compress_other_matrices = likelihood.compress_other_matrices
        self._append_to_other_matrices = likelihood.append_to_other_matrices


    def _set_feature_counts(self):
        self._feature_counts = self.Z.sum(axis=0)


    def _update_feature_counts(self, new, obj, feat):
        if new != self.Z[obj, latfeat]:
            self._feature_counts[obj, feat] += self.Z[obj, feat] - new

    def _set_log_counts_minus_obj(self):
        self._log_counts_minus_obj = np.log(self._feature_counts[None,:] - self.Z)


    def _update_log_counts_minus_obj(self, obj):
        self._log_counts_minus_obj[obj] = np.log(self._feature_counts - self.Z[obj])


    def _compute_log_posterior(self, val, obj, latfeat):
        log_likelihood = self._log_likelihood(val=val, obj=obj, latfeat=feat)

        return self._log_counts_minus_obj[obj, latfeat] + log_likelihood

    def _sample_val(self, obj, latfeat):
        logpost_on = self._compute_log_posterior(1, obj, latfeat)
        logpost_off = self._compute_log_posterior(0, obj, latfeat)

        prob = logpost_on / np.logaddexp(logpost_on, logpost_off)

        return sp.stats.bernoulli.rvs(prob)

    def _compress_Z(self):
        latent_features_gt_0 = self.feature_counts > 0

        self.Z = self.Z.compress(latent_features_gt_0, axis=1)
        self.feature_counts = self.feature_counts.compress(latent_features_gt_0)

        self._compress_other_matrices(latent_features_gt_0)

    def _append_new_latent_features(self, obj):
        num_of_new_latent_features = sp.stats.poisson.rvs(mu=self._poisson_param)

        if num_of_new_latent_features:
            new_latent_features = np.zeros([num_of_objects, num_of_new_latent_features])
            new_latent_features[obj] += 1

            self.Z = np.append(self.Z, new_latent_features, axis=1)
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
            self.Z[obj, latfeat] = new

            self._compress_Z()
            self._append_new_latent_features(obj)
            #self._reorder_Z()


    def _update_Z(self):

        for obj in np.arange(self.num_of_objects):
            self._update_Z_obj(obj)

