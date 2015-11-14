import os, abc
import numpy as np

from .data import *
from .interface import *
from .sampler import *
from .optimizer import *

class Model(object):

    def __init__(self, data, gamma=1., lmbda=1., alpha=1., beta=1.,
                 num_of_latfeats=2, nonparametric=False, 
                 sample_D=False, sample_Z=False, sample_B=False, blockiter=10, subiter=1,
                 proposalbandwidth_D=0.01, proposalbandwidth_Z=0.01, proposalbandwidth_B=0.01,
                 D=None, Z=None, B=None, pretrain_D=False, pretrain_ZB=False):

        self.data = data
        
        self.num_of_objects = data.num_of_objects
        self.num_of_obsfeats = data.num_of_obsfeats
        self.num_of_latfeats = num_of_latfeats

        self.nonparametric = nonparametric
                
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lmbda = lmbda
        
        ## set maximum block iterations
        self._blockiter = blockiter
        self._subiter = subiter
        
        self._initialize_D_inference(D, sample_D, proposalbandwidth_D)

        self._beta_posterior = BetaPosterior(D_inference=self._D_inference)
        
        self._initialize_ZB_inference(Z, B, sample_Z, sample_B, proposalbandwidth_Z, proposalbandwidth_B)

        if pretrain_D:
            self._pretrain()
        
                
    def _initialize_D_inference(self, D, sample_D, proposal_bandwidth):

        if sample_D:
            data_likelihood = PoissonGammaProductLikelihood(data=self.data,
                                                            gamma=self.gamma)

            self._D_inference = DSampler(D=D,
                                         likelihood=data_likelihood, 
                                         proposal_bandwidth=proposal_bandwidth)

            self._fit_D = self._D_inference.sample
            
        else:
            self._D_inference = DOptimizer(self.data, self.gamma, learning_rate=proposal_bandwidth, D=D)
            self._fit_D = lambda: self._D_inference.optimize(maxiter=self._blockiter)


    def _initialize_ZB_inference(self, Z, B, sample_Z, sample_B,
                                 proposal_bandwidth_Z, proposal_bandwidth_B):

        if sample_Z:
            Z = Z if isinstance(Z, np.ndarray) else (self.num_of_objects, self.num_of_latfeats)

            self._Z_inference = ZSampler(Z=Z,
                                         likelihood=self._beta_posterior,
                                         alpha=self.alpha,
                                         beta=self.beta,
                                         nonparametric=self.nonparametric)

            self._fit_Z = self._Z_inference.sample

        else:
            self._Z_inference = ZBOptimizer(self._D_inference,
                                            self.num_of_latfeats,
                                            self.alpha,
                                            self.beta,
                                            self.lmbda,
                                            learning_rate=proposal_bandwidth_Z,
                                            Z=Z,
                                            B=B)

            self._fit_Z = lambda: self._Z_inference.optimize(maxiter=self._blockiter,
                                                             subiter=self._subiter,
                                                             update_Z=True)
            
        if sample_B:
            B = B if isinstance(B, np.ndarray) else (self.num_of_latfeats, self.num_of_obsfeats)
            
            self._B_inference = BSampler(B=B,
                                         likelihood=self._beta_posterior,
                                         lmbda=lmbda,
                                         proposal_bandwidth=proposal_bandwidth_B)

            self._fit_B = self._B_inference.sample

        else:
            if sample_Z:            
                self._B_inference = ZBOptimizer(self._D_inference,
                                                self.num_of_latfeats,
                                                self.alpha,
                                                self.beta,
                                                self.lmbda,
                                                learning_rate=proposal_bandwidth_B,
                                                Z=self._Z_inference.get_param().astype(theano.config.floatX),
                                                B=B)
                self._fit_B = lambda: self._B_inference.optimize(maxiter=self._blockiter,
                                                                 subiter=self._subiter)

                Z, _ = self._beta_posterior.get_param()
                self._B_inference.set_Z(Z)

            else:
                self._B_inference = self._Z_inference
                self._fit_B = self._fit_Z
                        
    def _pretrain(self):
        ## initialize D with MLE estimate
        self._D_optimizer.optimize(maxiter=maxiter, objective_type='MLE')
        print 'MLE log-likelihood:', self._D_optimizer.compute_log_likelihood()

        np.savetxt(os.path.join(outdir, 'D_MLE.csv'), 
                   self._D_optimizer.get_param(),
                   #header=';'.join(self.data.obsfeats),
                   delimiter=';',
                   comments='',
                   fmt="%s")

            
        # if self._pretrain_ZB:
        #     if isinstance()
        #     self._ZB_inference.optimize(maxiter=maxiter, subiter=subiter, update_Z=True)

    
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
