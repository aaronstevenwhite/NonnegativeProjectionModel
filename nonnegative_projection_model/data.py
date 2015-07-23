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
    
    @classmethod
    def _default_obj_filter(obj):
        '''Filter predicate by whether it contains a copula or not'''
        
        return not re.match('^(be|become|are)\s', obj)
    
    def _load_data(self, fname, obj_filter=Data._default_obj_filter):
        '''
        Load data from CSV containing count data; extract object labels 
        from first column and observed feature labels from first row

        fname (str): path to data
        obj_filter (str -> bool): object filtering function
        '''

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
    
    def __init__(self, fname, obj_filter=Data._default_obj_filter):
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
    
    def __init__(self, fname, obj_filter=Data._default_obj_filter):
        '''
        Load data from CSV containing count data, extracting object labels 
        from first column and observed feature labels from first row

        fname (str): path to data
        obj_filter (str -> bool): object filtering function
        '''

        self._load_data(fname)

        self._initialize_obj_counts()
        
    def __iter__(self):
        return self

    def _initialize_obj_counts(self):
        '''
        Initialize variables for total number of times object seen, 
        number of times object-observed feature pair seen, and number 
        of times object-observed feature pair could be seen in the future
        '''

        self._data_obj_count = np.zeros(self._data.shape[0])
        self._unseen = self._data.astype(float)
        self._seen = np.zeros(self._data.shape)

        self._update_joint_prob()
        
    def _update_joint_prob(self):
        '''Update probability of seeing each object-observed feature pair'''        
        
        joint_prob = self._unseen / self._unseen.sum()
        self._joint_prob = joint_prob.flatten()
        
    def _sample_index(self):
        '''Sample a single object-observed feature pair'''        
        
        pair_index = np.random.choice(a=self._joint_prob.shape[0], 
                                      p=self._joint_prob)

        obj_index = pair_index / self._data.shape[1]
        obsfeat_index = pair_index / self._data.shape[0]
        
        self._unseen[obj_index, obsfeat_index] -= 1
        self._seen[obj_index, obsfeat_index] += 1

        self._update_joint_prob()

        return obj_index, obsfeat_index

    def next(self):
        try:
            assert self._unseen.sum() > 0
        except AssertionError:
            raise StopIteration

        datum = np.zeros(self._data.shape)
        datum[self._sample_index()] += 1

        return datum
