from setuptools import setup

setup(name='nonnegative_projection_model',
      version='0.1',
      description='Non-negative matrix factorization with gamma-poisson likelihood',
      url='http://github.com/aaronstevenwhite/NonnegativeProjectionModel',
      author='Aaron Steven White',
      author_email='aswhite@umd.edu',
      license='MIT',
      packages=['nonnegative_projection_model'],
      install_requires=['abc', 
                        'numpy', 
                        'scipy', 
                        'theano'],
      zip_safe=False)
