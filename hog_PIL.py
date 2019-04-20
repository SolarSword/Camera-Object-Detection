# -*- coding:utf-8 -*-
# using python library PIL for histogram...after a long time I discover that this API is not HOG
# :( very sad...very sad
# code could be much simpler, but still some problems 
import math
from PIL import Image as im
import cv2
import numpy as np 
#import matplotlib.pyplot as plt


def hog_similarity(lh,rh):
    '''Compute the similarity between two vectors

    Using a formula from a paper to compute the similarity
    
    Args:
        lh: Vector 1
        rh: Vector 2

    Returns:
        A float which describe the similarity 
        between two vectors. The closer this value is
        to 1, the more similar these two vectors are. 

    '''
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l,r)) for l, r in zip(lh, rh))/len(lh)

def calc_similarity(lh, rh):
    return hog_similarity(lh, rh)
    
def calc_similarity_split(lhs, rhs):
    '''Compute the similarity between two set of vectors

    Compute the similarity of two corresponding sub-vectors
    and return the average.

    Args: 
        lhs: Vector set 1, which is a list and whose elements 
            are vectors
        rhs: Vector set 2, same like lhs

    Returns:
        A float which is the average of the similarity
    '''
    similarities = []
    for i in range(len(lhs)):
        similarities.append(hog_similarity(lhs[i], rhs[i]))
    return sum(similarities) / len(lhs)


class HOG():
    '''
        This class is a package for HOG

    This class provides a set of methods to compute the 
    histogram of gradient of an image.

    Attributes:
        img: The image which is going to compute its histogram 
            of gradient.
        split_flag: A flag that shows whether to split image
            into cells and compute their histogram of gradient
            respectively.
        vector: The histogram of gradient of image.
        sub_images: A list which store the cells of image if 
            split_flag is True.
        vectors: The list which stores the histogram of gradient
            of the cells 
    '''
    
    def __init__(self, img, split_flag = False):
        '''
            Inits class
        '''
        self.img = img
        self.img = self.make_regular_image()
        if (split_flag):
            self.sub_images = self.split_image()
            self.vectors = self.get_hog_split()
        else:
            self.vector = self.get_hog()

   
    def make_regular_image(self,size = (256,256)):
        '''Make image to the same size  

        Args:
            size: Target size

        Returns:
            A PIL object which is resized image
        '''
        return self.img.resize(size).convert('RGB')
        

    def split_image(self,part_size = (64, 64)):
        '''Split image into small cells

        Args:
            part_size: The size of small cells

        Returns:
            A list which contants the small cells of image
        '''
        w, h = self.img.size
        pw, ph = part_size

        return [self.img.crop((i, j, i + pw, j + ph)).copy() for i in range(0, w, pw) for j in range(0, h ,ph)]

    def get_hog(self):
        # must regulization this vector into 0-127
        # because of the utf-8
        his = self.img.histogram()
        max_his = max(his)
        min_his = min(his)
        his = list(map(lambda x: int(127 * (max_his - x) / (max_his - min_his)), his))
        return his

    def get_hog_split(self):
        vectors = []
        for sub_image in self.sub_images:
            vectors.append(sub_image.histogram())
        return vectors
    
'''
im1 = im.open("1.jpg")
im2 = im.open("2.jpg")
hog1 = HOG(im1)
hog2 = HOG(im2)
print(calc_similarity(hog1.vector, hog2.vector))
print(type(hog1.vector))
'''

    

    