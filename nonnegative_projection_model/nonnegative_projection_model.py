import os, abc, warnings, re
import numpy as np
import scipy as sp
import theano
import theano.tensor as T

from scipy.optimize import minimize

##########
## data ##
##########

class Data(object):
    '''
    Base class containing methods for loading object-by-observed feature
    count data from CSV files with object labels in the first column and 
    observed feature labels in the first row
    '''
    
    @classmethod
    def _default_obj_filter(obj):
        '''Filter predicate by whether it contains a copula or not'''
        
        return not re.match('^(be|become|are)\s', obj)
    
    def _load_data(self, fname, obj_filter=Data._default_obj_filter):
        '''
        Load data from CSV containing count data, extracting object labels 
        from first column and observed feature labels from first row

        fname (str): path to data
        obj_filter (str -> bool): object filtering function
        '''

        ## load data from file
        data = np.loadtxt(fname, 
                          dtype=str, 
                          delimiter=',')

        ## extract counts and labels for objects and observed features
        objects = data[1:,0]
        obsfeats = data[0,1:]
        X = data[1:,1:].astype(int)

        ## vectorize the object filter function
        obj_filter_vectorized = np.vectorize(obj_filter)

        ## filter objects using the vectorized object filter
        objects_filtered = obj_filter_vectorized(objects)

        ## filter observed features by whether they non-zero counts
        obsfeats_filtered = X.sum(axis=0) > 0

        ## set data, object, and observed feature attributes
        self._objects = objects[objects_filtered]
        self._obsfeats = obsfeats[obsfeats_filtered]
        self._data = X[objects_filtered][:,obsfeats_filtered]
        
    def get_objects(self):
        '''Get object labels'''
        return self._objects

    def get_obsfeats(self):
        '''Get observed feature labels'''
        return self._obsfeats

    def get_data(self):
        '''Get count data'''
        return self._data

    def get_num_of_objects(self):
        '''Get number of objects (after filtering)'''
        return self.objects.shape[0]

    def get_num_of_obsfeats(self):
        '''Get number of (non-zero) observed features'''
        return self.obsfeats.shape[0]

    
class BatchData(Data):
    '''
    Wrapper for object-by-observed feature count data loaded from CSV;
    IO function _load_data is inherited from class Data
    '''
    
    def __init__(self, fname, obj_filter=Data._default_obj_filter):
        '''
        Load data from CSV containing count data, extracting object labels 
        from first column and observed feature labels from first row

        fname (str): path to data
        obj_filter (str -> bool): object filtering function
        '''
        
        self._load_data(fname, obj_filter)

        self._initialize_obj_counts()

    def _initialize_obj_counts(self):
        '''Count total number of times each object was seen'''
        
        self._data_obj_count = self._data.sum(axis=1)

    def get_obj_counts(self):
        '''Get total number of times each object was seen'''
        
        return self._data_obj_count

class IncrementalData(Data):
    '''
    Wrapper for object-by-observed feature count data loaded from CSV that 
    is iterable; incorporates a sampler for object-observed feature iterates
    IO function _load_data is inherited from class Data
    '''
    
    def __init__(self, fname, obj_filter=Data._default_obj_filter):
        '''
        Load data from CSV containing count data, extracting object labels 
        from first column and observed feature labels from first row

        fname (str): path to data
        obj_filter (str -> bool): object filtering function
        '''

        self._load_data(fname)

        self._initialize_obj_counts()
        
    def __iter__(self):
        return self

    def _initialize_obj_counts(self):
        '''
        Initialize variables for total number of times object seen, 
        number of times object-observed feature pair seen, and number 
        of times object-observed feature pair could be seen in the future
        '''

        self._data_obj_count = np.zeros(self._data.shape[0])
        self._unseen = self._data.astype(float)
        self._seen = np.zeros(self._data.shape)

        self._update_joint_prob()
        
    def _update_joint_prob(self):
        '''Update probability of seeing each object-observed feature pair'''        
        
        joint_prob = self._unseen / self._unseen.sum()
        self._joint_prob = joint_prob.flatten()
        
    def _sample_index(self):
        '''Sample a single object-observed feature pair'''        
        
        pair_index = np.random.choice(a=self._joint_prob.shape[0], 
                                      p=self._joint_prob)

        obj_index = pair_index / self._data.shape[1]
        obsfeat_index = pair_index / self._data.shape[0]
        
        self._unseen[obj_index, obsfeat_index] -= 1
        self._seen[obj_index, obsfeat_index] += 1

        self._update_joint_prob()

        return obj_index, obsfeat_index

    def next(self):
        try:
            assert self._unseen.sum() > 0
        except AssertionError:
            raise StopIteration

        datum = np.zeros(self._data.shape)
        datum[self._sample_index()] += 1

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
            self._data = data.get_data()
            self._data_obj_count = data.get_obj_counts()
            self._obj_range = data.get_obj_range()
            self._obsfeat_range = data.get_obsfeat_range()

        elif isinstance(data, IncrementalData):
            self._data_gen = data

    def link_prior(self, D_inference):
        self._D = D_inference.get_param()

    def get_data(self):
        return self._data.get_data()

    def _construct_indices(self, obj, obsfeat):
        obj = self._obj_range if obj is None else obj
        obsfeat = self._obsfeat_range if obsfeat is None else obsfeat

        return obj, obsfeat

    def _compute_ll_d_new(self, d_new, obj, obsfeat):
        log_numer = self._data[obj,obsfeat] * np.log(d_new)

        d_obj = np.insert(np.delete(self._D[obj], 
                                    obsfeat), 
                          obsfeat, 
                          d_new)

        log_denom_data = 1 + self._data_obj_count[obj]
        log_denom_D = np.log(self.gamma + d_obj.sum())
        log_denom = log_denom_data * log_denom_D

        return log_numer - log_denom

    def _compute_ll_D(self):
        log_numer = np.sum(self._data * np.log(self._D), axis=1)

        log_denom_data = 1 + self._data_obj_count[:,None]
        log_denom_D = np.log(self.gamma + self._D.sum(axis=1))
        log_denom = log_denom_data * log_denom_D

        return np.sum(log_numer - log_denom)


    def compute_log_likelihood(self, d_new=None, obj=None, obsfeat=None):
        if d_new is None:
            return self._compute_ll_D()
        else:
            return self._compute_ll_d_new(d_new, obj, obsfeat)


class BetaPosterior(Prior, Likelihood):

    def __init__(self, D_inference):
        self._D_inference = D_inference
        self._D = D_inference.get_param()

        self._B_inference = None

        self._Z = None
        self._B = None

    def link_prior(self, Z_inference=None, B_inference=None):
        ## don't need to save Z_inference
        self._B_inference = B_inference if B_inference != None else self._B_inference

        self._Z = Z_inference.get_param_Z() if Z_inference != None else self._Z
        self._B = B_inference.get_param_B() if B_inference != None else self._B

        if self._Z != None and self._B != None:
            self._ZB = np.dot(self._Z, self._B)
            self._tZB = theano.shared(self._ZB, name='ZB')

            self._D_inference.link_prior(self)

            self._previous_ll_z = np.zeros(2)
                
    def compress_and_append(self, latfeats_gt_0, num_of_new_features):
        self._B_inference.compress_and_append(self._Z, latfeats_gt_0, num_of_new_features)

    def get_data(self):
        return self._D

    def get_param(self):
        return self._Z, self._B

    def get_param_shared(self):
        return self._tZB

    def _compute_ll_z_old(self, obj, latfeat):
        z_old = self._Z[obj, latfeat]

        if latfeat != 0:
            prev_val = self._Z[obj, latfeat-1]

            return self._previous_ll_z[prev_val]

        else:
            z_obj = self._Z[obj]
            z_obj_B = np.dot(z_obj, self._B)
            z_obj_b_latfeat = z_old * self._B[latfeat]

            return np.sum(np.log(z_obj_B) + (z_obj_b_latfeat - 1)*np.log(self._D[obj]))
            
            
    def _compute_ll_z_new(self, obj, latfeat):
        z_new = np.abs(self._Z[obj,latfeat]-1)
            
        z_obj = np.insert(np.delete(self._Z[obj], 
                                    latfeat), 
                          latfeat, 
                          z_new)
        
        z_obj_B = np.dot(z_obj, self._B)
        z_obj_b_latfeat = z_new * self._B[latfeat]

        return np.sum(np.log(z_obj_B) + (z_obj_b_latfeat - 1)*np.log(self._D[obj]))

    def _compute_ll_z(self, obj, latfeat):
        z_old = self._Z[obj, latfeat]
        z_new = np.abs(z_old-1)

        ll_val = np.zeros(2)
        ll_val[z_old] = self._compute_ll_z_old(obj, latfeat)
        ll_val[z_new] = self._compute_ll_z_new(obj, latfeat)

        self._previous_ll_z = ll_val

        return ll_val 
        
    def _compute_ll_b_new(self, b_new, latfeat, obsfeat):
        b_obsfeat = np.insert(np.delete(self._B[:,obsfeat], 
                                        latfeat), 
                              latfeat, 
                              b_new)

        Zb_obsfeat = np.dot(self._Z, b_obsfeat)
        z_latfeat_b_obsfeat = b_new * self._Z[:,latfeat]

        ll = np.log(Zb_obsfeat) + (z_latfeat_b_obsfeat - 1)*np.log(self._D[:,obsfeat])

        return np.sum(ll)

    def compute_log_likelihood(self, z_new=None, b_new=None, obj=None, latfeat=None, obsfeat=None):
        if z_new != None:
            return self._compute_ll_z(obj, latfeat)

        elif b_new != None:
            return self._compute_ll_b_new(b_new, latfeat, obsfeat)
        
    def _update_log_prior_values(self):
        self._log_prior_values = (self._ZB - 1) * np.log(self._D)

    def compute_log_prior(self, d_new=None, obj=None, obsfeat=None):
        if d_new != None:
            return (self._ZB[obj, obsfeat] - 1) * np.log(d_new)

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
          Z (numpy.ndarray): object by latent feature association matrix
          likelihood (Likelihood): computes p(D|B,Z) and implements interfaces in ABC Likelihood
          alpha, beta (float): positive real number parameterizing Beta(alpha, beta) or IBP(alpha)
          nonparametric (bool): whether the prior should be considered nonparametric
        """

        self.alpha = alpha
        self.beta = beta

        self.nonparametric = nonparametric

        self._initialize_Z(Z, likelihood)

    def _initialize_Z(self, Z, likelihood):
        self._Z = Z
        #self._Z = np.ones(Z.shape)
        
        self._num_of_objects, self._num_of_latfeats = Z.shape

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

        #print obj, latfeat, logpost_off, logpost_on, prob
        
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
        new_loadings = np.random.exponential(1., shape).astype(theano.config.floatX)
        self._B = self._B.append(new_loadings, axis=0)




################
## optimizers ##
################

class Optimizer(object):

    def __init__(self, data, learning_rate=0.01):
        self._data = data.get_data().astype(theano.config.floatX)
        self._learning_rate = learning_rate
        self._num_of_rows, self._num_of_columns = self._data.shape
    
    @abc.abstractmethod
    def optimize(self):
        return

    @abc.abstractmethod
    def _construct_shared(self):
        return 


class DOptimizer(Optimizer):

    def __init__(self, X, gamma, learning_rate, D=None):
        self._data = X.astype(theano.config.floatX)

        if not isinstance(D, type(None)):
            self._D = D
        else:
            self._D = np.random.beta(1., 1., X.shape).astype(theano.config.floatX)
        
        self._tX = theano.shared(self._data, name='X')
        self._tD = theano.shared(self._D, name='D')
        
        self._likelihood_objective = self._construct_likelihood_objective(gamma)
        self._MLE_updater = self._construct_updater(self._likelihood_objective, learning_rate)

    def get_tD(self):
        return self._tD

    def _construct_likelihood_objective(self, gamma):
        tX, tD = self._tX, self._tD

        # if gamma:
        #     tgamma = theano.shared(theano.config.floatX(gamma))
        # else:
        #     tgamma = theano.shared(theano.config.floatX(1.))
            
        gamma_poisson_numer = T.sum(tX * T.log(tD), axis=1)
        gamma_poisson_denom = (1 + T.sum(tX, axis=1))*T.log(gamma+T.sum(tD, axis=1))
        
        likelihood_objective = T.sum(gamma_poisson_numer - gamma_poisson_denom)

        self.compute_log_likelihood = theano.function([], likelihood_objective)
        
        return likelihood_objective

    def _construct_prior_objective(self):
        tD = self._tD
        tZB = self._tZB + 1e-20

        return T.sum(T.log(tZB) + (tZB-1.)*T.log(tD))

    def _construct_posterior_objective(self):
        posterior_objective = self._prior_objective + self._likelihood_objective

        self.compute_log_posterior = theano.function([], posterior_objective)

        return posterior_objective

    def _construct_updater(self, objective, learning_rate):
        tD = self._tD

        gradient = T.tanh(T.grad(objective, tD))

        step_size_D = learning_rate * (T.minimum(tD, 1.-tD)-1e-20)

        return theano.function(inputs=[],
                               outputs=[],
                               updates={tD:tD+step_size_D*gradient},
                               name='update_D')

    def link_prior(self, prior, learning_rate=0.01):
        self._tZB = prior.get_param_shared()

        self._prior_objective = self._construct_prior_objective()
        self._posterior_objective = self._construct_posterior_objective()

        self._MAP_updater = self._construct_updater(self._posterior_objective, learning_rate)

    def get_prior_objective(self):
        return self._prior_objective

    def get_param(self):
        return self._D
    
    def optimize(self, maxiter=100, objective_type='MAP'):
        if objective_type=='MAP':
            updater = self._MAP_updater  
        else:
            updater = self._MLE_updater
            
        for i in np.arange(maxiter):
            updater()

        self._D = self._tD.get_value()

    def write(self, outfile):
        np.savetxt(outfile, 
                   self._D,
                   #header=';'.join(self.data.obsfeats),
                   delimiter=';',
                   #comments='',
                   fmt="%s")

        
class ZBOptimizer(Optimizer):

    def __init__(self, D_optimizer, num_of_latfeats, lmbda, learning_rate, Z=None, B=None):
        num_of_objects, num_of_obsfeats = D_optimizer.get_param().shape
        
        shape_Z = [num_of_objects, num_of_latfeats]
        shape_B = [num_of_latfeats, num_of_obsfeats]

        if not isinstance(Z, type(None)):
            self._Z = Z
        else:
            self._Z = np.random.exponential(1., shape_Z).astype(theano.config.floatX)

        if not isinstance(B, type(None)):
            self._B = B
        else:
            self._B = np.random.exponential(1., shape_B).astype(theano.config.floatX)

        self._ZB = np.dot(self._Z, self._B)
        
        self._tZ = theano.shared(self._Z, name='Z')
        self._tB = theano.shared(self._B, name='B')
        self._tZB = T.dot(self._tZ, self._tB)

        D_optimizer.link_prior(self, learning_rate)
        
        self._posterior_objective = self._construct_posterior_objective(D_optimizer, lmbda)
        self._MAP_updater_Z, self._MAP_updater_B = self._construct_updater(self._posterior_objective, learning_rate)

        
    def _construct_prior_objective(self, lmbda):
        prior_Z = T.sum(-lmbda*10*self._tZ)
        prior_B = T.sum(-lmbda*self._tB)

        self._compute_log_prior_Z = theano.function([], prior_Z)
        self._compute_log_prior_B = theano.function([], prior_B)
        self._compute_log_prior_ZB = theano.function([], prior_Z + prior_B)

        return prior_Z + prior_B
        
    def compute_log_prior(self, param='B'):
        if param == 'Z':
            return self._compute_log_prior_Z()
        elif param == 'B':
            return self._compute_log_prior_B()
        else:
            self._compute_log_prior_ZB()

    def _construct_likelihood_objective(self, D_optimizer):
        return D_optimizer.get_prior_objective()

    def _construct_posterior_objective(self, D_optimizer, lmbda):

        log_prior = self._construct_prior_objective(lmbda)
        log_likelihood = self._construct_likelihood_objective(D_optimizer)

        return log_prior + log_likelihood

    def _construct_updater(self, objective, learning_rate):        
        tZ = self._tZ
        tB = self._tB

        gradient_Z = T.tanh(T.grad(objective, tZ))
        gradient_B = T.tanh(T.grad(objective, tB))

        step_size_Z = learning_rate * T.minimum(tZ, T.abs_(T.log(tZ))+1.)
        step_size_B = learning_rate * T.minimum(tB, T.abs_(T.log(tB))+1.)

        update_Z = theano.function(inputs=[],
                                   outputs=[],
                                   updates={tZ:tZ+step_size_Z*gradient_Z},
                                   name='update_Z')
        update_B = theano.function(inputs=[],
                                   outputs=[],
                                   updates={tB:tB+step_size_B*gradient_B},
                                   name='update_B')

        return update_Z, update_B

    def set_Z(self, Z):
        self._Z = Z
        self._reset_shared()

    def set_B(self, B):
        self._B = B
        self._reset_shared()

    def _reset_shared(self):
        Z_noisy = np.where(self._Z == 0, 1e-20, 1.).astype(theano.config.floatX)
            
        self._tZ.set_value(Z_noisy)
        self._tB.set_value(self._B)
        
    def compress_and_append(self, latfeats_gt_0, num_of_new_features):
        self._B = self._B[latfeats_gt_0]

        shape = [num_of_new_features, self._B.shape[1]]
        new_loadings = np.random.exponential(1., shape).astype(theano.config.floatX)
        self._B = self._B.append(new_loadings, axis=0)

        self._reset_shared()
        
    def get_param(self):        
        return self._Z, self._B

    def get_param_Z(self):        
        return self._Z
    
    def get_param_B(self):        
        return self._B
    
    def get_param_shared(self):
        return self._tZB
    
    def optimize(self, maxiter=100, subiter=100, update_Z=False):
        updater_Z, updater_B = self._MAP_updater_Z, self._MAP_updater_B  

        if not update_Z:
            self._reset_shared()
        
        for i in np.arange(maxiter):
            if update_Z:
                for j in np.arange(subiter):
                    updater_Z()
            for j in np.arange(subiter):
                updater_B()

        if update_Z:
            self._Z = self._tZ.get_value()
            
        self._B = self._tB.get_value()
        self._ZB = np.dot(self._Z, self._B)
        
    def prepare_ZB(self):
        ## calculate max by feature
        Z_max = self._Z.max(axis=0)
        
        ## shift weights on B relative to largest weight
        ## for each feature    
        self._B = (self._B * Z_max[:,None]).astype(theano.config.floatX)

        ## normalize Z so max is 1 for each feature
        ## and threshold at .01 (should sluff off extra features when sampling started)
        Z_norm = self._Z / Z_max[None,:]
        self._Z = np.where(Z_norm>np.median(Z_norm, axis=0)[None,:], 1., 0.)

        ## reset the shared variables to their new values
        self._reset_shared()
        
        return self._Z
        
class JointOptimizer(Optimizer):

    def __init__(self, data, num_of_latfeats, gamma, lmbda, learning_rate=0.01,
                 D=None, Z=None, B=None, pretrain_D=False, pretrain_ZB=True):
        X = data.get_data()
        num_of_objects, num_of_obsfeats = X.shape

        self._pretrain_D = pretrain_D
        
        self._D_optimizer = DOptimizer(X, gamma, learning_rate, D=D)
        self._ZB_optimizer = ZBOptimizer(self._D_optimizer,
                                         num_of_latfeats,
                                         lmbda,
                                         learning_rate,
                                         Z=Z, B=B)

    def get_param(self):
        D = self._D_optimizer.get_param()
        Z, B = self._ZB_optimizer.get_param()

        return np.copy(D), np.copy(Z), np.copy(B)

    def prepare_Z(self):
        return self._ZB_optimizer.prepare_ZB()
    
    def get_optimizers(self):
        return self._D_optimizer, self._ZB_optimizer
    
    def optimize(self, maxiter, subiter):
        ## initialize D with MLE estimate
        if self._pretrain_D:
            self._D_optimizer.optimize(maxiter=maxiter, objective_type='MLE')
            print 'MLE log-likelihood:', self._D_optimizer.compute_log_likelihood()
        
        self._ZB_optimizer.optimize(maxiter=maxiter, subiter=subiter, update_Z=True)
        
        # for i in range(maxiter):
        #     print i, self._D_optimizer.compute_log_likelihood(), self._D_optimizer.compute_log_likelihood()+self._ZB_optimizer.compute_log_prior('Z') +self._ZB_optimizer.compute_log_prior('B')
        #     self._ZB_optimizer.optimize(maxiter=1, subiter=subiter, update_Z=True)
        #     self._D_optimizer.optimize(maxiter=1, objective_type='MAP')
        
    def write(self, outdir):
        D, Z, B = self.get_param()
        
        np.savetxt(os.path.join(outdir, 'D_MLE.csv'), 
                   D,
                   #header=';'.join(self.data.obsfeats),
                   delimiter=';',
                   #comments='',
                   fmt="%s")

        np.savetxt(os.path.join(outdir, 'B_MAP.csv'), 
                   B,
                   #header=';'.join(self.data.obsfeats),
                   delimiter=';',
                   #comments='',
                   fmt="%s")

        np.savetxt(os.path.join(outdir, 'Z_MAP.csv'), 
                   Z,
                   #header=';'.join(self.data.obsfeats),
                   delimiter=';',
                   #comments='',
                   fmt="%s")



#######################
## fitting procedure ##
#######################

class Model(object):

    def __init__(self, data, num_of_latfeats, gamma, lmbda, alpha, beta, nonparametric,
                 sample_D, sample_B, distributionproposalbandwidth, featloadingsproposalbandwidth,
                 initializationmaxiter, initializationsubiter, samplermaxiter, samplersubiter,
                 D=None, Z=None, B=None):

        self.data = data

        self.num_of_latfeats = num_of_latfeats if num_of_latfeats else 10
        
        joint_optimizer = JointOptimizer(data=data,
                                         num_of_latfeats=num_of_latfeats,
                                         gamma=gamma,
                                         lmbda=lmbda,
                                         D=D, Z=Z, B=B, pretrain_D=isinstance(D, type(None)))

        joint_optimizer.optimize(maxiter=initializationmaxiter, subiter=initializationsubiter)

        self._joint_optimizer = joint_optimizer
        
        self._initialize_D_inference(data, joint_optimizer, sample_D, gamma,
                                     samplermaxiter, distributionproposalbandwidth)

        self._beta_posterior = BetaPosterior(D_inference=self._D_inference)
        
        self._initialize_Z_inference(joint_optimizer, alpha, beta, nonparametric)
        self._initialize_B_inference(joint_optimizer, sample_B, lmbda,
                                     samplermaxiter, samplersubiter,
                                     featloadingsproposalbandwidth)
        
    def _initialize_D_inference(self, data, joint_optimizer, sample_D, gamma,
                                maxiter, proposal_bandwidth):

        initial_D, _, _ = joint_optimizer.get_param()
        D_optimizer, _ = joint_optimizer.get_optimizers()

        self._initial_D = initial_D
        
        if sample_D:
            data_likelihood = PoissonGammaProductLikelihood(data=data,
                                                            gamma=gamma)

            self._D_inference = DSampler(D=initial_D,
                                         likelihood=data_likelihood, 
                                         proposal_bandwidth=proposal_bandwidth)

            self._fit_D = self._D_inference.sample
        else:
            self._D_inference = D_optimizer
            self._fit_D = lambda: self._D_inference.optimize(maxiter=maxiter)


    def _initialize_Z_inference(self, joint_optimizer, alpha, beta, nonparametric):
        _, self._initial_Z, _ = joint_optimizer.get_param()
        initial_Z = joint_optimizer.prepare_Z()
        
        self._Z_inference = ZSampler(Z=initial_Z,
                                     likelihood=self._beta_posterior,
                                     alpha=alpha,
                                     beta=beta,
                                     nonparametric=nonparametric)

        self._fit_Z = self._Z_inference.sample

    def _initialize_B_inference(self, joint_optimizer, sample_B, lmbda, maxiter, subiter, proposal_bandwidth):
        _, _, initial_B = joint_optimizer.get_param()
        _, ZB_optimizer = joint_optimizer.get_optimizers()

        self._initial_B = initial_B
        
        if sample_B:
            self._B_inference = BSampler(B=initial_B,
                                         likelihood=self._beta_posterior,
                                         lmbda=lmbda,
                                         proposal_bandwidth=proposal_bandwidth)
            self._fit_B = self._B_inference.sample

        else:            
            self._B_inference = ZB_optimizer
            self._fit_B = lambda: self._B_inference.optimize(maxiter=maxiter, subiter=subiter)

            Z, _ = self._beta_posterior.get_param()
            self._B_inference.set_Z(Z)

            self._beta_posterior.link_prior(B_inference=ZB_optimizer)

    
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
        
    def _save_samples(self, itr, burnin, thinning):
        self._D_samples[(itr-burnin)/thinning] = self._D_inference.get_param()
        self._Z_samples[(itr-burnin)/thinning] = self._Z_inference.get_param()
        self._B_samples[(itr-burnin)/thinning] = self._B_inference.get_param()


    def compute_log_likelihood(self):
         return float(self._D_inference.compute_log_likelihood())

    def compute_log_posterior(self):
        posterior_D = self._D_inference.compute_log_posterior()
        prior_Z = self._Z_inference.compute_log_prior()
        prior_B = self._B_inference.compute_log_prior()

        return posterior_D + prior_Z + prior_B
    
    def fit(self, iterations, burnin, thinning):
        self._initialize_samples(iterations, burnin, thinning)

        for preitr in np.arange(burnin):
            print 'Z', preitr
            self._fit_Z()

        self._fit_B()
        self._fit_D()
        
        for itr in np.arange(iterations):
            print 'iteration', itr
            print 'likelihood:', self.compute_log_likelihood()
            print 'posterior:', self.compute_log_posterior(), '\n'

            self._fit()

            if itr >= burnin and not itr % thinning:
                self._save_samples(itr, burnin, thinning)

    def write(self, outdir):

        np.savetxt(os.path.join(outdir, 'D'+str(self.num_of_latfeats)+'.csv'), 
                   self._D_samples[-1],
                   header=';'.join(self.data.obsfeats),
                   delimiter=';',
                   comments='',
                   fmt="%s")
        
        np.savetxt(os.path.join(outdir, 'B'+str(self.num_of_latfeats)+'.csv'), 
                   self._B_samples[-1][1],
                   header=';'.join(self.data.obsfeats),
                   delimiter=';',
                   comments='',
                   fmt="%s")

        np.savetxt(os.path.join(outdir, 'Z'+str(self.num_of_latfeats)+'.csv'), 
                   self._Z_samples[-1],
                   delimiter=';',
                   comments='',
                   fmt="%s")

