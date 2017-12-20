from __future__ import division, print_function, absolute_import

import tensorflow as tf
from nn.load_params import convShape
import numpy as np
import logging
def convLayer_FT(inpTensor, kShape, s, name):
    inpTensor = tf.cast(inpTensor, tf.float32)
    
    with tf.variable_scope(name+'_wb'):
        w = tf.get_variable(
                dtype='float32',
                shape=kShape,
                initializer=tf.truncated_normal_initializer(
                         stddev=0.1, seed=6752
                ),
                name="convWeight",
                trainable=True
        )
    
        b = tf.get_variable(
                dtype='float32',
                shape=[kShape[-1]],
                initializer=tf.constant_initializer(1),
                name="convBias",
                trainable=True
    
        )

    tf.summary.histogram("convWeights", w)
    tf.summary.histogram("convbias", b)
    act = tf.nn.conv2d(inpTensor, w, [1, s, s, 1], padding='VALID', name=name) + b
    
    return act

def batchNorm_FT(inpTensor, numOUT, axis=[0,1,2], name=None):
    with tf.variable_scope(name+'_bn'):
        beta = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(0.0),
                name="beta",
                trainable=True
        )
        gamma = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(1.0),
                name="gamma",
                trainable=True)
        batchMean, batchVar = tf.nn.moments(inpTensor, axes=axis, name="moments")
        bn = tf.nn.batch_normalization(inpTensor, mean=batchMean, variance=batchVar,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=1e-5, name=name)
        return bn
        
class Inception_FT():
    def __init__(self):
        '''
            The model in Fine tune is pretty similar to that of The Inception model.
            In the (mini) Inception model we use the pre-trained parameters for the 5a and 5b
            layers. Here we keep the Inception network fix until the 5a layer and train our
            network for 5a and 5b layer (fine tune) for out set of faces.
        '''
        pass

    def inception_1x1(self, X, cnv1s, name='5a'):
        conv_name = 'inception_%s_1x1_conv' % str(name)
        bn_name = 'inception_%s_1x1_bn' % str(name)
        kShape = np.array(convShape[conv_name])
        kShape = [kShape[2], kShape[3], kShape[1], kShape[0]]
        X_1x1 = convLayer_FT(X, kShape, s=cnv1s, name=conv_name)
        X_1x1 = batchNorm_FT(X_1x1, kShape[-1], axis=[0,1,2], name=bn_name)
        X_1x1 = tf.nn.relu(X_1x1)
        logging.info('inception_1x1_FT: Chain 1: shape = %s', str(X_1x1.shape))
        return X_1x1


    def inception_3x3(self, X, cnv1s, cnv2s, padTD, padLR , name='5a'):
        conv1_name = 'inception_%s_3x3_conv1' % str(name)
        bn1_name = 'inception_%s_3x3_bn1' % str(name)
        conv2_name = 'inception_%s_3x3_conv2' % str(name)
        bn2_name = 'inception_%s_3x3_bn2' % str(name)
        k1_shape = np.array(convShape[conv1_name])
        k1_shape = [k1_shape[2], k1_shape[3], k1_shape[1], k1_shape[0]]
        k2_shape = np.array(convShape[conv2_name])
        k2_shape = [k2_shape[2], k2_shape[3], k2_shape[1], k2_shape[0]]


        X_3x3 = convLayer_FT(X, k1_shape, s=cnv1s, name=conv1_name)
        X_3x3 = batchNorm_FT(X_3x3, k1_shape[-1], axis=[0, 1, 2], name=bn1_name)
        X_3x3 = tf.nn.relu(X_3x3)
        
        # Zero Padding
        X_3x3 = tf.pad(X_3x3, paddings=[[0,0], [padTD[0], padTD[1]], [padLR[0], padLR[1]], [0, 0]])
        X_3x3 = convLayer_FT(X_3x3, k2_shape, s=cnv2s, name=conv2_name)
        X_3x3 = batchNorm_FT(X_3x3, k2_shape[-1], axis=[0, 1, 2], name=bn2_name)
        X_3x3 = tf.nn.relu(X_3x3)
        logging.info('inception_3x3 Chain 2: shape = %s ', str(X_3x3.shape))
        return X_3x3

    def inception_pool(self, X, cnv1s, padTD, padLR, poolSize, poolStride,
                       poolType, name='5a'):
        conv_name = 'inception_%s_pool_conv' % str(name)
        bn1_name = 'inception_%s_pool_bn' % str(name)
        k1_shape = np.array(convShape[conv_name])
        k1_shape = [k1_shape[2], k1_shape[3], k1_shape[1], k1_shape[0]]
        
        # Chain 4
        if poolType == 'avg':
            X_pool = tf.layers.average_pooling2d(X, pool_size=poolSize, strides=poolStride,
                                                 data_format='channels_last')
        elif poolType == 'max':
            X_pool = tf.layers.max_pooling2d(X, pool_size=poolSize, strides=poolStride,
                                             data_format='channels_last')
        else:
            X_pool = X
    
        X_pool = convLayer_FT(X_pool, k1_shape, s=cnv1s, name=conv_name)
        
        # X_pool = tf.layers.batch_normalization(X_pool, axis=1, epsilon=1e-5, name=bn1_name)
        X_pool = batchNorm_FT(X_pool, numOUT=k1_shape[-1], axis=[0,1,2], name=bn1_name)
        X_pool = tf.nn.relu(X_pool)
        X_pool = tf.pad(X_pool, paddings=[[0, 0], [padTD[0], padTD[1]],
                                          [padLR[0], padLR[1]], [0, 0]])
        logging.info('inception_pool Chain 4: shape = %s ', str(X_pool.shape))
        return X_pool

