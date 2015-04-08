from nonnnegative_projection_model import *
from warnings import warn

##################
## argument parser
##################

## initialize parser
parser = argparse.ArgumentParser(description='Fit non-negative projection model.')

## file handling
parser.add_argument('--data', 
                    type=str, 
                    default='./frame_counts.csv')
parser.add_argument('--output', 
                    type=str, 
                    default='./model_fits/full_gibbs')

## model hyperparameters
parser.add_argument('--nonparametric', 
                    nargs='?', 
                    const=True, 
                    default=False)
parser.add_argument('--featurenum', 
                    type=int, 
                    default=10)

parser.add_argument('--featureform',
                    type=str, 
                    choices=['discrete', 'continuous'], 
                    default='discrete')
parser.add_argument('--loadingprior', 
                    type=str, 
                    choices=['exponential', 'laplace'], 
                    default='exponential')

parser.add_argument('--featuresparsity', 
                    type=float, 
                    default=1.)
parser.add_argument('--loadingsparsity', 
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
    warn('''if parameters are loaded, they will not be pre-optimized\ 
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
