import abc
import numpy as np
import scipy as sp
import theano

from scipy.optimize import minimize

##############
## likelihoods
##############

class Likelihood(object):

    ## binary matrix
    
    @abc.abstractmethod
    def set_Z(self, Z):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_log_likelihood_Z(self, Z, obj, latfeat):
        raise NotImplementedError

    @abc.abstractmethod
    def compress_other_matrices(self, latfeat_bool):
        raise NotImplementedError

    @abc.abstractmethod
    def append_to_other_matrices(self, num_of_new_latent_features):
        raise NotImplementedError

    ## loading matrix

    @abc.abstractmethod
    def set_B(self, B):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_log_likelihood_B(self, B, latfeat, obsfeat):
        raise NotImplementedError

    ## distribution matrix
    @abc.abstractmethod
    def compute_log_likelihood_D(self, D, latfeat=None, obsfeat=None):
        raise NotImplementedError


class PoissonBetaLikelihood(Likelihood):

    def __init__(self, data_fname, gamma, delta, optimizer='L-BFGS-B'):
        ## Gamma parameters
        self.gamma = gamma
        self.delta = delta

        self.optimizer = optimizer

        self._load_X(data_fname)
        self._initialize_D()

    def _load_X(self, fname):
        data = np.loadtxt(fname, 
                          dtype=str, 
                          delimiter=',')

        self.objects = data[1:,0]
        self.features = data[0,1:]
        self.X = data[1:,1:].astype(int)

        self._X_obj_count = self.X.sum(axis=1)

    def get_X_obj_count(self, obj, obsfeat):
        return self._X_obj_count[obj, obsfeat]

    def _log_likelihood(self, D, i=[], j=[]):
        i = range(self.X.shape[0]) if not len(i) else i
        j = range(self.X.shape[1]) if not len(j) else j
            
        log_numer = np.sum(self.X[i,j] * np.log(D[i,j]), axis=1)

        log_denom_X = self.gamma + self._X_obj_count[i]
        log_denom_D = np.log(self.delta + D[i].sum(axis=1))
        log_denom = log_denom_X * log_denom_D

        return np.sum(log_numer - log_denom)

    def compute_log_likelihood_D(self, D, latfeat=None, obsfeat=None):
        raise NotImplementedError

    def _log_likelihood_jacobian(self, D, i=[], j=[]):
        i = range(self.X.shape[0]) if not len(i) else i
        j = range(self.X.shape[1]) if not len(j) else j

        first_term = self.X[i,j] / D[i,j]

        second_term_numer = (self.gamma + self._X_obj_count[i])[:,None]
        second_term_denom = (self.delta + D[i].sum(axis=1))[:,None]
        second_term = second_term_numer / second_term_denom

        return first_term - second_term

    def _initialize_D(self):
        initial_D = self.X/self.X.sum(axis=1)[:,None]

        if self.optimizer:
            ll_flattened = lambda D: -self._log_likelihood(D.reshape(self.X.shape))
            ll_jac_flattened = lambda D: -self._log_likelihood_jacobian(D.reshape(self.X.shape)).flatten()

            bounds = [(10e-20,1.-10e-20)]*np.prod(self.X.shape)

            self.D = minimize(fun=ll_flattened,
                              x0=initial_D.flatten(),
                              jac=ll_jac_flattened,
                              bounds=bounds,
                              method=self.optimizer).reshape(self.X.shape)

        else:
            self.D = initial_D

    def _compute_log_likelihood(self, D=None):
        if D != None:
            return self._log_likelihood(D)

        else:
            return self._log_likelihood(self.D)

    def set_Z(self, Z):
        self.Z = Z

    def set_B(self, B):
        self.B = B

    def set_D(self, D):
        self.D = D

    def get_D(self):
        return self.D

    def compute_log_likelihood_Z(self, Z, obj, latfeat):
        self.set_Z(Z)
        raise NotImplementedError

    def compute_log_likelihood_B(self, B, latfeat, obsfeat):
        self.set_B(B)
        raise NotImplementedError

    def compress_other_matrices(self, latfeat_bool):
        raise NotImplementedError

    def append_to_other_matrices(self, num_of_new_latent_features):
        raise NotImplementedError

###########
## samplers
###########

class ZSampler(object):

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


class BSampler():

    def __init_(self):
        raise NotImplementedError


class DSampler():
    
    def __init_(self, likelihood, proposal_bandwidth=1.):
        self._proposal_bandwidth = proposal_bandwidth
        self._initialize_likelihood(likelihood)
        self._initialize_log_posterior_values()

    def _compute_ZB(self):
        self.ZB = np.dot(self.Z, self.B)

    def _initialize_likelihood(self, likelihood):
        self.Z = likelihood.get_Z()
        self.B = likelihood.get_B()
        self.D = likelihood.get_D()

        self._compute_ZB()

        self._log_likelihood = likelihood.compute_log_likelihood_D

    def _initialize_log_posterior_values(self):
        log_like = self._log_posterior_values = self._log_likelihood(self.D)
        log_prior = (self.ZB[obj, obsfeat] - 1) * np.log(self.D)

        self._log_posterior_values = log_like + log_prior

    def _proposer(self, obj, obsfeat):
        d_mn_old = self.D[obj, obsfeat]
        window_size = np.min([d_mn_old, 1-d_mn_old]) / self._proposal_bandwidth
        
        return np.random.uniform(low=d_mn_old-window_size, 
                                 high=d_mn_old+window_size)

    def _log_posterior(self, obj, obsfeat, d_mn=None):
        d_mn = d_mn if d_mn!=None else self.D[obs, obsfeat] 

        log_like = self._log_likelihood(d_mn, obj, obsfeat)
        log_prior = (self.ZB[obj, obsfeat] - 1) * np.log(d_mn)

        return log_like + log_prior

    def _acceptance_prob(self, proposal, obj, obsfeat):
        self._log_posterior(proposal)
