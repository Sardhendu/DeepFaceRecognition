from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf


class Preprocessing():
    '''
        Preprocessing in images are done per image, hence it is a good idea to create a separate computation graph
        for Preprocessing such that the graph is iteratively fed the input image one after another pertaining to a
        batch.
    '''
    
    def __init__(self):
        pass
    
    def randomCrop(self, imageIN):
        pass
    
    def randomFlip(self, imageIN):
        # Given an image this operation may or may not flip the image
        return tf.image.random_flip_left_right(imageIN)
    
    def addRandBrightness(self, imageIN):
        # Add random brightness
        return tf.image.random_brightness(imageIN, max_delta=63)
    
    def addRandContrast(self, imageIN):
        return tf.image.random_contrast(imageIN, lower=0.2, upper=1.8)
    
    def standarize(self, imageIN):
        return tf.image.per_image_standardization(imageIN)
    
    def preprocessImageGraph(self, imageSize, numChannels):
        """
        :param imageSize:   The size of image
        :param numChannels: The number of channels
        :return:  The distorted image
        """
        '''
            Normally the inputs have dtype unit8 (0-255), We however take the input as float32 because we perform
            operations like brightness, contrast and whitening that doest arithmatic operation which many make the
            pixels value as floating point.
        '''
        imageIN = tf.placeholder(dtype=tf.float32,
                                 shape=[imageSize[0], imageSize[1], numChannels],
                                 name="Preprocessor-variableHolder")
        
        # Add random contrast
        imageOUT = self.randomFlip(imageIN)
        imageOUT = self.addRandBrightness(imageOUT)
        imageOUT = self.addRandContrast(imageOUT)
        imageOUT = self.standarize(imageOUT)
        
        return dict(imageIN=imageIN,
                    imageOUT=imageOUT)