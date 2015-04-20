# NonnegativeProjectionModel

This package implements samplers and optimizers for the computational model of syntactic bootstrapping proposed by White (2015). Both parametric and nonparametric priors on feature matrices are available (see Griffiths & Gharamani 2006, 2011 for derivations of these priors).

## Design

The package is broken into five classes of interacting objects. The `Data` class and its subclasses implement input interfaces. The `Likelihood` and `Prior` classes and subclasses compute values of conditional PMFs/PDFs and implement interfaces between `Sampler` and `Optimizer` objects. The `Sampler` class and subclasses implement latent parameter sampling. The `Optimizer` class and subclasses implement latent parameter (bounded) optimization for the subset of latent parameters that are continuous. The `Model` class implements joint sampling and provides various output interfaces, including graphing and model checking.  