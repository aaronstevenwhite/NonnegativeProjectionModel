from setuptools import setup

setup(name='nonnegative_projection_model',
      version='0.1dev',
      description='Non-negative matrix factorization with gamma-poisson likelihood',
      url='http://github.com/aaronstevenwhite/NonnegativeProjectionModel',
      author='Aaron Steven White',
      author_email='aswhite@jhu.edu',
      license='MIT',
      packages=['nonnegative_projection_model'],
      install_requires=['numpy', 
                        'scipy', 
                        'theano',
                        'pandas'],
      zip_safe=False)
