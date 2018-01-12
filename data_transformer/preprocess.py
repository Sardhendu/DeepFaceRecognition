from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import config

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
        logging.info('Initialing random horizontal flip')
        if config.preprocess_seed_idx == len(config.seed_arr) -1:
            config.preprocess_seed_idx = 0
        # Given an image this operation may or may not flip the image
        return tf.image.random_flip_left_right(imageIN,
                                               seed=config.seed_arr[config.preprocess_seed_idx])
    
    def addRandBrightness(self, imageIN):
        logging.info('Adding random brightness')
        if config.preprocess_seed_idx == len(config.seed_arr) -1:
            config.preprocess_seed_idx = 0
        # Add random brightness
        return tf.image.random_brightness(imageIN, max_delta=63,
                                          seed=config.seed_arr[config.preprocess_seed_idx])
    
    def addRandContrast(self, imageIN):
        logging.info('Adding random Contrast')
        if config.preprocess_seed_idx == len(config.seed_arr) -1:
            config.preprocess_seed_idx = 0
        return tf.image.random_contrast(imageIN, lower=0.2, upper=1.8,
                                        seed=config.seed_arr[config.preprocess_seed_idx])
    
    def standarize(self, imageIN):
        logging.info('Standarizing the image')
        return tf.image.per_image_standardization(imageIN)
    
    def preprocessImageGraph(self, imageShape):
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
        logging.info('PREPROCESSING THE DATASET ..........')
        imageIN = tf.placeholder(dtype=tf.float32,
                                 shape=[imageShape[0], imageShape[1], imageShape[2]],
                                 name="Preprocessor-variableHolder")
        
        # Add random contrast
        imageOUT = self.randomFlip(imageIN)
        imageOUT = self.addRandBrightness(imageOUT)
        imageOUT = self.addRandContrast(imageOUT)
        imageOUT = self.standarize(imageOUT)
        
        return dict(imageIN=imageIN,
                    imageOUT=imageOUT)