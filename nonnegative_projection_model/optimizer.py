import os, abc
import numpy as np
import scipy as sp
import theano
import theano.tensor as T

from .data import BatchData, IncrementalData 

class Optimizer(object):

    def __init__(self, data, learning_rate=0.01):
        self._data = data.data.astype(theano.config.floatX)
        self._learning_rate = learning_rate
        self._num_of_rows, self._num_of_columns = self._data.shape
    
    @abc.abstractmethod
    def optimize(self):
        return

    @abc.abstractmethod
    def _construct_shared(self):
        return 


class DOptimizer(Optimizer):

    def __init__(self, data, gamma, learning_rate, D=None):
        self._data = data

        if D is not None:
            self._D = D
        else:
            self._D = 0.0001*np.ones(self._data.shape).astype(theano.config.floatX)
    
        self._tAUX = theano.shared(np.log(self._D) - np.log(1-self._D), name='tAUX')
        self._tD = 1 / (1 + T.exp(-self._tAUX))
        
        if isinstance(data, BatchData):
            self._online = False
            self._tX = theano.shared(self._data.data, name='X')
        elif isinstance(data, IncrementalData):
            self._online = True
            self._tX = theano.shared(self._data.next(), name='X')
        
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
        tAUX = self._tAUX

        gradient = T.tanh(T.grad(objective, tAUX))

        return theano.function(inputs=[],
                               outputs=[],
                               updates={tAUX:tAUX+learning_rate*gradient},
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
            if self._online:
                self._tX.set_value(self._data.next())
                
            updater()

        self._D = self._tD.eval()

    def write(self, outfile):
        np.savetxt(outfile, 
                   self._D,
                   #header=';'.join(self.data.obsfeats),
                   delimiter=';',
                   #comments='',
                   fmt="%s")

        
class ZBOptimizer(Optimizer):

    def __init__(self, D_optimizer, num_of_latfeats, alpha, beta,
                 lmbda, learning_rate, Z=None, B=None):
        num_of_objects, num_of_obsfeats = D_optimizer.get_param().shape
        
        shape_Z = [num_of_objects, num_of_latfeats]
        shape_B = [num_of_latfeats, num_of_obsfeats]

        if not isinstance(Z, type(None)):
            self._Z = Z
        else:
            self._Z = np.random.beta(1., 1., shape_Z).astype(theano.config.floatX)
            
        if not isinstance(B, type(None)):
            self._B = B
        else:
            self._B = np.random.exponential(1., shape_B).astype(theano.config.floatX)

        self._ZB = np.dot(self._Z, self._B)

        ## REMOVE HACK
        AUX = np.log(self._Z) - np.log(1-self._Z)
        AUX[10,0] = float(10)
        AUX[10,1:] = float(-10) 
        ## REMOVE HACK
        
        self._tAUX = theano.shared(AUX, name='tAUX')
        self._tZ = 1 / (1 + T.exp(-self._tAUX))
        
        self._tB = theano.shared(self._B, name='B')
        self._tZB = T.dot(self._tZ, T.abs_(self._tB))

        D_optimizer.link_prior(self, learning_rate)
        
        self._posterior_objective = self._construct_posterior_objective(D_optimizer, alpha, beta, lmbda)
        self._MAP_updater_Z, self._MAP_updater_B = self._construct_updater(self._posterior_objective, learning_rate)

        
    def _construct_prior_objective(self, alpha, beta, lmbda):
        prior_Z = T.sum((alpha-1)*T.log(self._tZ) +\
                        (beta-1.)*T.log(1-self._tZ))
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

    def _construct_posterior_objective(self, D_optimizer, alpha, beta, lmbda):

        log_prior = self._construct_prior_objective(alpha, beta, lmbda)
        log_likelihood = self._construct_likelihood_objective(D_optimizer)

        return log_prior + log_likelihood

    def _construct_updater(self, objective, learning_rate):        
        tAUX = self._tAUX
        tB = self._tB

        gradient_Z = T.grad(objective, tAUX)
        gradient_B = T.grad(objective, tB)

        ## REMOVE HACK
        x = learning_rate*np.ones(self._Z.shape, dtype=theano.config.floatX)
        x[10,0] = 0.
        x[10,1] = 0.
        ## REMOVE HACK

        print x
        
        update_Z = theano.function(inputs=[],
                                   outputs=[],
                                   updates={tAUX:tAUX+x*gradient_Z},
                                   name='update_Z')
        update_B = theano.function(inputs=[],
                                   outputs=[],
                                   updates={tB:tB+learning_rate*gradient_B},
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
            self._Z = self._tZ.eval()
            
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
