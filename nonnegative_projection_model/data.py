import re
import numpy as np

'''
This module provides classes for data representation; these classes provide
interfaces to the raw cooccurrence data contains in a CSV file
'''

class Data(object):
    '''
    Base class containing methods for loading object-by-observed feature
    count data from CSV files with object labels in the first column and 
    observed feature labels in the first row
    '''
    
    @staticmethod
    def _default_obj_filter(obj):
        '''Filter predicate by whether it contains a copula or not'''
        
        return not re.match('^(be|become|are)\s', obj)
    
    def _load_data(self, fname, obj_filter):
        '''
        Load data from CSV containing count data; extract object labels 
        from first column and observed feature labels from first row

        fname (str): path to data
        obj_filter (str -> bool or NoneType): object filtering function
        '''

        if obj_filter is None:
            obj_filter = self.__class__._default_obj_filter
        
        ## load data from file
        data = np.loadtxt(fname, 
                          dtype=str, 
                          delimiter=',')

        ## extract counts and labels for objects and observed features
        objects = data[1:,0]
        obsfeats = data[0,1:]
        X = data[1:,1:].astype(int)

        ## vectorize the object filter function
        obj_filter_vectorized = np.vectorize(obj_filter)
        
        ## filter objects using the vectorized object filter
        objects_filtered = obj_filter_vectorized(objects)
        
        ## filter observed features by whether they non-zero counts
        obsfeats_filtered = X.sum(axis=0) > 0

        ## set data, object, and observed feature attributes
        self._objects = objects[objects_filtered]
        self._obsfeats = obsfeats[obsfeats_filtered]
        self._data = X[objects_filtered][:,obsfeats_filtered]

    @property
    def objects(self):
        return self._objects

    @property
    def obsfeats(self):
        return self._obsfeats

    @property
    def data(self):
        '''Get count data'''
        return self._data

    @property
    def shape(self):
        '''Get count data shape'''
        return self._data.shape
            
    @property
    def num_of_objects(self):
        return self.objects.shape[0]

    @property
    def num_of_obsfeats(self):
        return self.obsfeats.shape[0]

    
class BatchData(Data):
    '''
    Wrapper for object-by-observed feature count data loaded from CSV;
    IO function _load_data is inherited from class Data
    '''
    
    def __init__(self, fname, obj_filter=None):
        '''
        Load data from CSV containing count data, extracting object labels 
        from first column and observed feature labels from first row

        fname (str): path to data
        obj_filter (str -> bool): object filtering function
        '''
        
        self._load_data(fname, obj_filter)

        self._initialize_obj_counts()

    def _initialize_obj_counts(self):
        '''Count total number of times each object was seen'''
        
        self._obj_counts = self._data.sum(axis=1)

    @property
    def obj_counts(self):        
        return self._obj_counts

class IncrementalData(Data):
    '''
    Wrapper for iterable object-by-observed feature count data loaded from CSV; 
    produces one-hot object-observed feature matrices stochastically by sampling
    without replacement; IO function _load_data is inherited from class Data
    '''
    
    def __init__(self, fname, obj_filter=None):
        '''
        Load data from CSV containing count data, extracting object labels 
        from first column and observed feature labels from first row

        fname (str): path to data
        obj_filter (str -> bool): object filtering function
        '''

        self._load_data(fname, obj_filter)

        self._initialize_sample_sequence()
        
    def __iter__(self):
        return self

    def _initialize_sample_sequence(self):
        '''
        Initialize probability of seeing each object-observed feature pair
        and construct sample sequence
        '''

        joint_prob = (self._data.astype(float) / self._data.sum()).ravel()

        print joint_prob
        
        pair_indices = np.random.choice(a=joint_prob.shape[0],
                                        size=self._data.sum()*100,
                                        p=joint_prob)

        pair_indices = np.column_stack(np.unravel_index(pair_indices,
                                                        dims=self._data.shape))
        
        self._sample_gen = ([i, j] for i, j in pair_indices)
                
    def next(self):
        sample_ind = self._sample_gen.next()
        datum = np.zeros(self._data.shape)
        datum[sample_ind[0], sample_ind[1]] += 1

        return datum
