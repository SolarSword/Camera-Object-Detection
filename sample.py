# -*- coding: utf-8 -*-
# the class of Sample
import random
import math

import numpy as np 
class Sample():
    '''Sample class

    Assuming that the sample is one-dimentional vector.
    Also this class can be used as the centroid or sum of samples
    '''
    def __init__(self, vector = np.array([]), label = None):
        if(type(vector) == type(np.array([]))):
            self.vector = vector
            self.length = len(vector)
        else:
            raise ValueError('The type of np_vector is not numpy.ndarray.')
        self.label = label

    def set_vector(self, np_vector):
        '''To set the np_vector as the self.vector

        Args:
            np_vector: the vector to be set, must be a numpy array
        '''
        if(type(np_vector) == type(np.array([]))):
            self.vector = np_vector
            self.length = len(np_vector)
        else:
            raise ValueError('The type of np_vector is not numpy.ndarray.')

    def set_label(self, label):
        self.label = label

    def __add__(self, other):
        '''Reconstract operator +

        This kind of reconstract methods are almost the same, 
        so latter comments are omitted

        Args:
            other: another Sample object

        Return:
            a new Sample object with the vector of this add result
        '''
        if(not(type(self) == type(other))):
            raise ValueError('The type is different.')
        elif(self.length == other.length  ):
            temp_object = Sample()
            temp_vector = self.vector + other.vector
            temp_object.set_vector(temp_vector)
            return temp_object
        else:
            raise ValueError('The vector length is different.') 

    def __len__(self):
        return self.length

    def __truediv__(self, other):
        temp_object = Sample()
        temp_vector = self.vector / other
        temp_object.set_vector(temp_vector)
        return temp_object

    def distance(self, other):
        '''Compute the distence between two Samples

        Args:
            other: another Sample object

        Return:
            The euclidean distance between two Samples' vectors. But this 
            distance is averaged over self.length.
        '''
        if(not(type(self) == type(other))):
            raise ValueError('The type is different.')
        elif(self.length == other.length  ):
            square_sum = 0
            for i in range(self.length):
                square_sum += (self.vector[i] - other.vector[i]) * (self.vector[i] - other.vector[i])
            return math.sqrt(square_sum) / self.length
        else:
            raise ValueError('The vector length is different.') 
