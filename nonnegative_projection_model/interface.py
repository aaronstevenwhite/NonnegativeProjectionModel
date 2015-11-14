import os, abc, warnings, re
import numpy as np
import scipy as sp
import theano
import theano.tensor as T

class Likelihood(object):
    '''
    Abstract base class for classes that compute the likelihood
    of some parameter(s) given some data
    '''
    
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_log_likelihood(self, parameter, **kwargs):
        '''Compute the log-likelihood of parameter given data'''
        
        return

class Prior(object):
    '''
    Abstract base class for classes that compute the prior 
    probability of some parameter(s) given some hyperparameters
    '''
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def compute_log_prior(self, parameter, **kwargs):
        '''Compute the prior probability of parameter given data'''

        return

class PoissonGammaProductLikelihood(Likelihood):
    '''
    Likelihood of D: p(X|D; gamma) = int dg p(X|D, g)p(g; gamma)
    '''
    
    def __init__(self, data, gamma=1.):
        '''
        Initialize likelihood of D given subclass of Data and
        Gamma distribution shape parameter gamma (aka alpha or k)

        data (subclass of Data): count data
        gamma (float): Gamma distribution shape parameter
        '''
        
        self.gamma = gamma
        self._data = data

    @property
    def data(self):
        return self._data.data

    @property
    def obj_counts(self):
        return self._data.obj_counts
         
    def _compute_ll_d_new(self, D, d_new, obj, obsfeat):
        '''
        Compute the log likelihood that the value of D[obj,obsfeat]
        is d_new

        D (np.array): unit-valued object-by-observed feature matrix
        d_new (float): proposed unit value of D[obj,obsfeat]
        obj (int): row index
        obsfeat (int): column index
        '''
        
        log_numer = self.data[obj,obsfeat] * np.log(d_new)

        d_obj = np.insert(np.delete(D[obj], obsfeat), 
                          obsfeat, 
                          d_new)

        log_denom_data = 1 + self.obj_counts[obj]
        log_denom_D = np.log(self.gamma + d_obj.sum())
        log_denom = log_denom_data * log_denom_D

        return log_numer - log_denom

    def _compute_ll_D(self, D):
        '''
        Compute the total log likelihood of D

        D (np.array): unit-valued object-by-observed feature matrix
        '''
        
        log_numer = np.sum(self.data * np.log(D), axis=1)

        log_denom_data = 1 + self.obj_count[:,None]
        log_denom_D = np.log(self.gamma + D.sum(axis=1))
        log_denom = log_denom_data * log_denom_D

        return np.sum(log_numer - log_denom)

    def compute_log_likelihood(self, D=None, d_new=None, obj=None, obsfeat=None):
        '''
        Compute either (i) the total log likelihood of D or (ii) the
        log likelihood that the value of D[obj,obsfeat] is d_new
        to compute the total LL of D, only pass this method D
        to the LL that d_new = D[obj,obsfeat], pass this method all parameters

        D (np.array): unit-valued object-by-observed feature matrix
        d_new (float): proposed unit value of D[obj,obsfeat]
        obj (int): row index
        obsfeat (int): column index
        '''

        if d_new is not None:
            return self._compute_ll_d_new(D, d_new, obj, obsfeat)
        else:
            return self._compute_ll_D(D)


class BetaPosterior(Prior, Likelihood):

    def __init__(self, D_inference):
        '''
        Initialize posterior of D given sampler or optimizer (D_inference)
        Samplers/optimizers for Z and B must be linked 
        
        D_inference (subclass of Sampler or Optimizer): sampler or optimizer
                                                        for D matrix
        '''

        self._D_inference = D_inference
        self._D = D_inference.get_param()
        
        self._Z_inference = None
        self._B_inference = None

        self._Z = None
        self._B = None

    @property
    def data(self):
        return self._D_inference.param
        
    def link_prior(self, Z_inference=None, B_inference=None):
        ## don't need to save Z_inference but will anyway
        self._Z_inference = Z_inference if Z_inference is not None else self._Z_inference
        self._B_inference = B_inference if B_inference is not None else self._B_inference

        self._Z = Z_inference.get_param_Z() if Z_inference is not None else self._Z
        self._B = B_inference.get_param_B() if B_inference is not None else self._B

        if self._Z is not None and self._B is not None:
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
        if z_new is not None:
            return self._compute_ll_z(obj, latfeat)

        elif b_new is not None:
            return self._compute_ll_b_new(b_new, latfeat, obsfeat)
        
    def _update_log_prior_values(self):
        self._log_prior_values = (self._ZB - 1) * np.log(self._D)

    def compute_log_prior(self, d_new=None, obj=None, obsfeat=None):
        if d_new is not None:
            return (self._ZB[obj, obsfeat] - 1) * np.log(d_new)

        else:
            self._update_log_prior_values()
            return self._log_prior_values[obj, obsfeat]
