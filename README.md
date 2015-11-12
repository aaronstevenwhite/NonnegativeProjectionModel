# NonnegativeProjectionModel

This package implements samplers and optimizers for the computational model of syntactic bootstrapping proposed in White 2015. Both parametric and nonparametric priors on feature matrices are available (see Griffiths & Ghahramani 2005, 2011 for derivations of these priors). Both batch and online inference are also available. Online inference is implemented as the analogue of local MAP (Anderson 1991, Sanborn et al. 2010) for a featural model (compare the particle filter-based approach in Wood & Griffiths 2006).

## Design

The package is broken into five classes of interacting objects. The `Data` class and its subclasses implement input interfaces (including a stochastic interface for online inference). The `Likelihood` and `Prior` classes and subclasses compute values of conditional PMFs/PDFs and implement interfaces between `Sampler` and `Optimizer` objects. The `Sampler` class and subclasses implement latent parameter sampling. The `Optimizer` class and subclasses implement latent parameter (bounded) optimization for the subset of latent parameters that are continuous. The `Model` class implements joint sampling and provides various output interfaces, including graphing and model checking.  

## References

Griffiths, T., & Ghahramani, Z. (2005). Infinite latent feature models and the Indian buffet process.

Griffiths, T. L., & Ghahramani, Z. (2011). The indian buffet process: An introduction and review. The Journal of Machine Learning Research, 12, 1185-1224.

Sanborn, A. N., Griffiths, T. L., & Navarro, D. J. (2010). Rational approximations to rational models: Alternative algorithms for category learning. Psychological Review, 117 (4), 1144-1167.

White, A.S. (2015). Information and incrementality in syntactic bootstrapping. PhD Thesis, University of Maryland, College Park.

Wood, F., & Griffiths, T. L. (2006). Particle filtering for nonparametric Bayesian matrix factorization. In Advances in Neural Information Processing Systems (pp. 1513-1520).
