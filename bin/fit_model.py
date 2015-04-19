import warnings, argparse

from nonnegative_projection_model import *

##################
## argument parser
##################

## initialize parser
parser = argparse.ArgumentParser(description='Fit non-negative projection model.')

## file handling
parser.add_argument('--data', 
                    type=str, 
                    default='../bin/frame_counts.csv')
parser.add_argument('--output', 
                    type=str, 
                    default='../bin/model_fits/full_gibbs')

## model form
parser.add_argument('--featurenum', ## if set to 0, nonparametric prior used 
                    type=int, 
                    default=0)
parser.add_argument('--featureform',
                    type=str, 
                    choices=['discrete', 'continuous'], 
                    default='discrete')
parser.add_argument('--loadingprior', 
                    type=str, 
                    choices=['exponential', 'laplace'], 
                    default='exponential')

## model hyperparameters
parser.add_argument('--alpha', ## feature success
                    type=float, 
                    default=1.)
parser.add_argument('--beta', ## feature failure; won't be used if nonparametric
                    type=float, 
                    default=1.)
parser.add_argument('--lambda', ## feature loading sparsity 
                    type=float, 
                    default=1.)
parser.add_argument('--gamma', ## feature loading sparsity 
                    type=float, 
                    default=1.)


## parameter initialization
parser.add_argument('--loadobjfeatures', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadfeatloadings', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loaddistributions', 
                    nargs='?',
                    const=True,
                    default=False)

parser.add_argument('--optimizeobjfeatures', 
                    type=str, 
                    choices=['', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'], 
                    default='')
parser.add_argument('--optimizefeatloadings', 
                    type=str, 
                    choices=['', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'], 
                    default='')
parser.add_argument('--optimizedistributions', 
                    type=str, 
                    choices=['', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'], 
                    default='')

## sampler parameters
parser.add_argument('--iterations', 
                    type=int, 
                    default=110000)
parser.add_argument('--burnin', 
                    type=int, 
                    default=10000)
parser.add_argument('--thinning', 
                    type=int, 
                    default=100)

parser.add_argument('--distributionproposalbandwidth', 
                    type=float, 
                    default=1.)
parser.add_argument('--featloadingsproposalbandwidth', 
                    type=float, 
                    default=1.)

parser.add_argument('--samplefeatloadings', 
                    nargs='?',
                    const=False,
                    default=True)
parser.add_argument('--sampledistributions', 
                    nargs='?',
                    const=False,
                    default=True)

## parse arguments
args = parser.parse_args()

##################
## check arguments
##################

try:
    assert args.loadobjfeatures != args.optimizeobjfeatures
    assert args.loadfeatloadings != args.optimizefeatloadings
    assert args.loaddistributions != args.optimizedistributions
except AssertionError:
    warnings.warn('''if parameters are loaded, they will not be pre-optimized\ 
            using MLE, though they will be optimized instead of sampled\ 
            during model fit''')

try:
    assert args.samplefeatloadings or args.optimizefeatloadings
    assert args.sampledistributions or args.optimizedistributions
except AssertionError:
    raise ValueError, '''if parameters are not sampled, they must be optimized;\ 
                         please choose a constrained optimizer (e.g. L-BFGS-B)\
                         from those made available in scipy.optimize'''


##########
## fitting
##########

data_likelihood = PoissonGammaProductLikelihood(data_fname=args.data, 
                                                gamma=args.gamma)

D_sampler = DSampler(likelihood=data_likelihood, 
                     proposal_bandwidth=args.distributionproposalbandwidth, 
                     optimizer=args.optimizedistributions)

beta_posterior = BetaPosterior(D_inference=D_sampler)

Z_sampler = ZSampler(likelihood=beta_posterior,
                     alpha=args.alpha,
                     num_of_latfeats=args.featurenum)

# B_sampler = BSampler(likelihood=beta_posterior,
#                      lam=args.alpha,
#                      num_of_latfeats=Z_sampler.get_num_of_latfeats())


# model = FullGibbs(D_sampler, Z_sampler, B_sampler)
# model.fit(args.iterations, args.burnin, args.thinning)
