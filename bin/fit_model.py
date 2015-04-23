import warnings, argparse

from nonnegative_projection_model import *
from nmf import initialize_D_Z_B

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
parser.add_argument('--nonparametric',
                    type=bool, 
                    default=False)
parser.add_argument('--featurenum',
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
parser.add_argument('--lmbda', ## feature loading sparsity 
                    type=float, 
                    default=1.)
parser.add_argument('--gamma', ## feature loading sparsity 
                    type=float, 
                    default=1.)

## parameter initialization
parser.add_argument('--loadobjectfeatures', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loadfeatureloadings', 
                    nargs='?',
                    const=True,
                    default=False)
parser.add_argument('--loaddistributions', 
                    nargs='?',
                    const=True,
                    default=False)

## optimization parameters
parser.add_argument('--optimizeobjectfeatures', 
                    type=bool, 
                    default=False)
parser.add_argument('--optimizefeatureloadings', 
                    type=bool, 
                    default=False)
parser.add_argument('--optimizedistributions', 
                    type=bool, 
                    default=False)

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

## metropolis-hastings proposer parameters
parser.add_argument('--distributionproposalbandwidth', 
                    type=float, 
                    default=1.)
parser.add_argument('--featloadingsproposalbandwidth', 
                    type=float, 
                    default=1.)

## parse arguments
args = parser.parse_args()

##################
## check arguments
##################

try:
    assert args.nonparametric ^ args.featurenum
except AssertionError:
    warnings.warn('''both nonparametric and featurenum were set;\
                     featurenum will only be used for initialization;\
                     the number of latent features will be sampled''')    

try:
    assert args.nonparametric != (args.beta == 1.)
except AssertionError:
    warnings.warn('''sampler has been set to nonparametric and\ 
                     beta has been set to a value different from 1;\
                     beta will be ignored (Pitman-Yor is not implemented)''')

try:
    assert args.loadobjectfeatures != args.optimizeobjectfeatures
    assert args.loadfeatureloadings != args.optimizefeatureloadings
    assert args.loaddistributions != args.optimizedistributions
except AssertionError:
    warnings.warn('''parameters will be loaded but not pre-optimized;\
                     they will be optimized instead of sampled\ 
                     during model fit''')

##########
## fitting
##########

print 'loading data'

data = BatchData(data_fname=args.data)

num_of_latfeats = args.featurenum if args.featurenum else 100

initial_D, initial_Z, initial_B = initialize_D_Z_B(data.get_data(),
                                                   num_of_latfeats,
                                                   maxiter=100,
                                                   subiter=100)

data_likelihood = PoissonGammaProductLikelihood(data=data,
                                                gamma=args.gamma)

print 'initializing D sampler'

D_sampler = DSampler(D=initial_D,
                     likelihood=data_likelihood, 
                     proposal_bandwidth=args.distributionproposalbandwidth)

beta_posterior = BetaPosterior(D_inference=D_sampler)

print 'initializing Z sampler'

Z_sampler = ZSampler(Z=initial_Z,
                     likelihood=beta_posterior,
                     alpha=args.alpha,
                     nonparametric=args.nonparametric)

print 'initializing B sampler'

B_sampler = BSampler(B=initial_B,
                     likelihood=beta_posterior,
                     lmbda=args.lmbda,
                     proposal_bandwidth=args.featloadingsproposalbandwidth)

# model = FullGibbs(D_sampler, Z_sampler, B_sampler)
# model.fit(args.iterations, args.burnin, args.thinning)
