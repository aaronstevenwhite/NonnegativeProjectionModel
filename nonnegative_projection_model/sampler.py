import os, abc
import numpy as np
import scipy as sp

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

        likelihood.link_prior(D_inference=self)

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
        if d_new is not None:
            return self._compute_lp_d_new(d_new, obj, obsfeat)

        elif D is not None:
            return self._compute_lp_D(D)

        else:
            return self._log_posterior_values[obj, obsfeat]

    def _proposer(self, obj, obsfeat):
        window_size = self._proposal_window[obj, obsfeat]
        
        return np.random.uniform(low=self._D[obj,obsfeat]-window_size, 
                                 high=self._D[obj,obsfeat]+window_size)

    def _log_proposal_prob(self, obj, obsfeat, proposal=None):
        if proposal is None:
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
          Z (numpy.ndarray or tuple): object by latent feature association matrix
          likelihood (Likelihood): computes p(D|B,Z) and implements interfaces in ABC Likelihood
          alpha, beta (float): positive real number parameterizing Beta(alpha, beta) or IBP(alpha)
          nonparametric (bool): whether the prior should be considered nonparametric
        """

        self.alpha = alpha
        self.beta = beta

        self.nonparametric = nonparametric

        self._initialize_Z(Z, likelihood)

    def _initialize_Z(self, Z, likelihood):
        self._Z = Z if isinstance(Z, np.ndarray) else np.random.binomial(1, 0.5, Z)

        print self._Z
        
        self._num_of_objects, self._num_of_latfeats = self._Z.shape

        self._obj_range = range(self._num_of_objects)
        self._latfeat_range = range(self._num_of_latfeats)

        likelihood.link_prior(Z_inference=self)
        self._log_likelihood = likelihood.compute_log_likelihood

        if self.nonparametric:
            ## need to compress on initialization because optimization might return 0 features
            self._compress_and_append_to_others = likelihood.compress_and_append
            self._smoothing = np.array([self._num_of_objects-1., 0.])
            self._poisson_param = float(alpha/self._num_of_objects)
        else:
            self._smoothing = np.array([self.beta+self._num_of_objects-1., self.alpha])

        self._feature_counts = self._Z.sum(axis=0)

        feature_count_minus_obj = self._feature_counts[None,:] - self._Z
        counts_smoothed = self._smoothing[:,None,None] + np.array([-1, 1])[:,None,None] * feature_count_minus_obj

        self._log_prior_values = np.log(counts_smoothed)

    def _update_feature_counts(self, obj_change):
        self._feature_counts += obj_change
            
    def _update_log_prior_values(self, obj):
        feature_count_minus_obj = self._feature_counts - self._Z[obj]
        counts_smoothed = self._smoothing[:,None] + np.array([-1, 1])[:,None] * feature_count_minus_obj

        self._log_prior_values[:,obj] = np.log(counts_smoothed)

    def _compute_log_posterior(self, z_new, obj, latfeat):
        log_likelihood = self._log_likelihood(z_new=z_new, obj=obj, latfeat=latfeat)

        return self._log_prior_values[:, obj, latfeat] + log_likelihood

    def _compress(self):
        latfeats_gt_0 = self.feature_counts > 0

        self._Z = self._Z.compress(latfeats_gt_0, axis=1)
        self.feature_counts = self.feature_counts.compress(latfeats_gt_0)

        return latfeats_gt_0
        
    def _append_new_latfeats(self, obj):
        num_of_new_latfeats = sp.stats.poisson.rvs(mu=self._poisson_param)

        if num_of_new_latfeats:
            new_latfeats = np.zeros([self._num_of_objects, num_of_new_latfeats])
            new_latfeats[obj] += 1

            self._Z = np.append(self._Z, new_latfeats, axis=1)
            self.feature_counts = np.append(self.feature_counts, 
                                            np.ones(num_of_new_latfeats))

            self._latfeat_range = range(self._Z.shape[1])

        return num_of_new_latfeats
            
    def get_num_of_latfeats(self):
        return self._Z.shape[1]

    def get_param(self):
        return self._Z

    def get_param_Z(self):
        return self._Z
    
    def compute_log_prior(self):
        V = self._num_of_objects

        if self.nonparametric:
            alpha = 0.
            beta = 1.
        else:
            alpha = self.alpha
            beta = self.beta

        log_numer1 = sp.special.gammaln(self.alpha+self._feature_counts)
        log_numer2 = sp.special.gammaln(self.beta+V-self._feature_counts)
        log_denom = sp.special.gammaln(self.alpha+self.beta+V)

        return np.sum(log_numer1 + log_numer2 - log_denom)

    def _sample(self, obj, latfeat):
        z_new = np.abs(self._Z[obj, latfeat]-1)
        
        logpost_off, logpost_on = self._compute_log_posterior(z_new, obj, latfeat)
        
        logpost_on = logpost_on if logpost_on != -np.inf else -1e20
        logpost_off = logpost_off if logpost_off != -np.inf else -1e20
        
        prob = np.exp(logpost_on - np.logaddexp(logpost_on, logpost_off))
        
        return sp.stats.bernoulli.rvs(prob)

    def sample(self):
        for obj in self._obj_range:
            obj_change = np.zeros(self._num_of_latfeats)
            for latfeat in self._latfeat_range:
                z_new = self._sample(obj, latfeat)
                obj_change[latfeat] = z_new - self._Z[obj, latfeat]
                self._Z[obj, latfeat] = z_new
                
            self._update_feature_counts(obj_change)
            self._update_log_prior_values(obj)
            
            if self.nonparametric:
                latfeats_gt_0 = self._compress()
                num_of_new_latfeats = self._append_new_latfeats()
                self._compress_and_append_to_others(latfeats_gt_0, num_of_new_latfeats)


class BSampler(Sampler):

    def __init__(self, B, likelihood, lmbda, proposal_bandwidth):
        self.lmbda = lmbda

        self._initialize_B(B, likelihood)
        self._initialize_ranges()
        self._initialize_proposer(proposal_bandwidth)

    def _initialize_B(self, B, likelihood):
        self._log_likelihood = likelihood.compute_log_likelihood
        self._B = B

        likelihood.link_prior(B_inference=self)

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
        if b_new is not None:
            return self._compute_lp_b_new(b_new, obj, obsfeat)

        elif B is not None:
            return self._compute_lp_B(B)

        else:
            return self._log_posterior_values[latfeat, obsfeat]

    def _proposer(self, latfeat, obsfeat):
        window_size = self._proposal_window[latfeat, obsfeat]
        
        return np.random.uniform(low=self._B[latfeat,obsfeat]-window_size, 
                                 high=self._B[latfeat,obsfeat]+window_size)

    def _log_proposal_prob(self, latfeat, obsfeat, proposal=None):
        if proposal is None:
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

    def get_param_B(self):        
        return self._B
    
    def compress_and_append(self, latfeat_gt_0, num_of_new_features):
        self._B = self._B[latfeat_gt_0]
        shape = [num_of_new_features, self._num_of_obsfeats]
        new_loadings = np.random.exponential(1., shape).astype(np.float)
        self._B = self._B.append(new_loadings, axis=0)
