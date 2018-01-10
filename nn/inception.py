import numpy as np
import tensorflow as tf
import logging

from nn.load_params import convShape

def convLayer(inpTensor, w, b, s, isTrainable, name):
    inpTensor = tf.cast(inpTensor, tf.float32)
    
    if not isTrainable:
        weights = tf.constant(w, name="weights", dtype=tf.float32)
        bias = tf.constant(b, name='biases', dtype=tf.float32)
    else:
        '''
            This part would endure that the first value for the variable weight and bias is taken
            from the inception net and the weights are updated based in training
            Why? Because the pretrained weights are good for initialization of weights.
        '''
        with tf.variable_scope(name + '_wb'):
            weights = tf.Variable( w, dtype='float32', name='weight', trainable=True)
            bias = tf.Variable(b, dtype='float32', name="biases", trainable=True)
    
    tf.summary.histogram("convWeights", weights)
    tf.summary.histogram("convBias", bias)
    act = tf.nn.conv2d(inpTensor, weights, [1, s, s, 1], padding='VALID', name=name) + bias
    
    return act

def batchNorm(inpTensor, mean, var, gamma, beta, numOUT=None, isTrainable=False, name=None):
    '''
    :param inpTensor:    Input Tensor to Normalize
    :param mean:         The exponential weighted mean for each channel obtained from training
                          sample.
    :param var:          The exponential weighted variance for each channel obtained from training
                          sample.
    :param gamma:        The trained scale (can be thought as weight for Batch Normalization)
    :param beta:         The trained offset (can be thought as bias for Batch Normalization)
    :param isTrainable:
    :param name:
    :return:
    
    Note: The mean, bar, gamma, beta should be 1D Tensor or an array of channel Size
          and the input should be in the form of [m, h, w, channels]
    '''
    inpTensor = tf.cast(inpTensor, tf.float32)
    if not isTrainable:
        m = tf.constant(mean, name="Exponential_weighted_mean", dtype=tf.float32)
        v = tf.constant(var, name="Exponential_weighted_variance", dtype=tf.float32)
        b = tf.constant(beta, name="Offset", dtype=tf.float32)
        w = tf.constant(gamma, name='Scale', dtype=tf.float32)
        bn = tf.nn.batch_normalization(inpTensor, mean=m, variance=v, offset=b, scale=w,
                                       variance_epsilon=1e-5, name=name)
    else:
        axis = [0,1,2]
        with tf.variable_scope(name + '_bn'):
            beta = tf.get_variable(
                    dtype='float32',
                    shape=[numOUT],
                    initializer=tf.constant_initializer(0.0),
                    name="beta_offset",
                    trainable=True
            )
            gamma = tf.get_variable(
                    dtype='float32',
                    shape=[numOUT],
                    initializer=tf.constant_initializer(1.0),
                    name="gamma_scale",
                    trainable=True)
            batchMean, batchVar = tf.nn.moments(inpTensor, axes=axis, name="moments")
            bn = tf.nn.batch_normalization(inpTensor, mean=batchMean, variance=batchVar,
                                           offset=beta, scale=gamma,
                                           variance_epsilon=1e-5, name=name)
    
    return bn


def activation(X, type='relu'):
    if type == 'relu':
        return tf.nn.relu(X)
    elif type == 'sigmoid':
        return tf.nn.sigmoid(X)

class Inception():
    def __init__(self, params, trainable=False):
        self.params = params
        self.trainable = trainable
    
    def inception_1x1(self, X, cnv1s=1, name='3a'):
        conv_name = 'inception_%s_1x1_conv'%str(name)
        bn_name = 'inception_%s_1x1_bn'%str(name)
        k1_shape = np.array(convShape[conv_name])
        k1_shape = [k1_shape[2], k1_shape[3], k1_shape[1], k1_shape[0]]
        
        X_1x1 = convLayer(X, self.params[conv_name]['w'], self.params[conv_name]['b'], s=cnv1s,
                          isTrainable=self.trainable, name=conv_name)
        # X_1x1 = tf.layers.batch_normalization(X_1x1, axis=1, epsilon=1e-5, name=bn_name)
        X_1x1 = batchNorm(X_1x1,
                          mean=self.params[bn_name]['m'], var=self.params[bn_name]['v'],
                          gamma=self.params[bn_name]['w'], beta=self.params[bn_name]['b'],
                          numOUT=k1_shape[-1], isTrainable=self.trainable, name=bn_name)
        X_1x1 = tf.nn.relu(X_1x1)
        logging.info('inception_1x1: Chain 1: shape = %s', str(X_1x1.shape))
        return X_1x1
    
    def inception_3x3(self, X, cnv1s=1, cnv2s=1, padTD=(1,1), padLR=(1,1), name='3a'):
        # Chain 2
        conv1_name = 'inception_%s_3x3_conv1'%str(name)
        bn1_name = 'inception_%s_3x3_bn1'%str(name)
        conv2_name = 'inception_%s_3x3_conv2'%str(name)
        bn2_name = 'inception_%s_3x3_bn2'%str(name)
        k1_shape = np.array(convShape[conv1_name])
        k1_shape = [k1_shape[2], k1_shape[3], k1_shape[1], k1_shape[0]]
        k2_shape = np.array(convShape[conv2_name])
        k2_shape = [k2_shape[2], k2_shape[3], k2_shape[1], k2_shape[0]]
        
        X_3x3 = convLayer(X, self.params[conv1_name]['w'], self.params[conv1_name]['b'], s=cnv1s,
                          isTrainable=self.trainable, name=conv1_name)
        # X_3x3 = tf.layers.batch_normalization(X_3x3, axis=1, epsilon=1e-5, name=bn1_name)
        X_3x3 = batchNorm(X_3x3,
                          mean=self.params[bn1_name]['m'], var=self.params[bn1_name]['v'],
                          gamma=self.params[bn1_name]['w'], beta=self.params[bn1_name]['b'],
                          numOUT=k1_shape[-1], isTrainable=self.trainable, name=bn1_name)
        X_3x3 = tf.nn.relu(X_3x3)
        
        X_3x3 = tf.pad(X_3x3, paddings=[[0, 0], [padTD[0], padTD[1]], [padLR[0], padLR[1]], [0, 0]])  #####  Zeropadding
        X_3x3 = convLayer(X_3x3, self.params[conv2_name]['w'], self.params[conv2_name]['b'],
                          s=cnv2s, isTrainable=False, name=conv2_name)
        # X_3x3 = tf.layers.batch_normalization(X_3x3, axis=1, epsilon=1e-5, name=bn2_name)
        X_3x3 = batchNorm(X_3x3,
                          mean=self.params[bn2_name]['m'], var=self.params[bn2_name]['v'],
                          gamma=self.params[bn2_name]['w'], beta=self.params[bn2_name]['b'],
                          numOUT=k2_shape[-1], isTrainable=self.trainable, name=bn2_name)
        X_3x3 = tf.nn.relu(X_3x3)
        logging.info('inception_3x3 Chain 2: shape = %s', str(X_3x3.shape))
        return X_3x3
    
    def inception_5x5(self, X, cnv1s=1, cnv2s=1, padTD=(2,2), padLR=(2,2), name='3a'):
        # Chain 3
        conv1_name = 'inception_%s_5x5_conv1' % str(name)
        bn1_name = 'inception_%s_5x5_bn1' % str(name)
        conv2_name = 'inception_%s_5x5_conv2' % str(name)
        bn2_name = 'inception_%s_5x5_bn2' % str(name)
        k1_shape = np.array(convShape[conv1_name])
        k1_shape = [k1_shape[2], k1_shape[3], k1_shape[1], k1_shape[0]]
        k2_shape = np.array(convShape[conv2_name])
        k2_shape = [k2_shape[2], k2_shape[3], k2_shape[1], k2_shape[0]]
        
        X_5x5 = convLayer(X, self.params[conv1_name]['w'], self.params[conv1_name]['b'], s=cnv1s,
                          isTrainable=self.trainable, name=conv1_name)
        # X_5x5 = tf.layers.batch_normalization(X_5x5, axis=1, epsilon=1e-5, name=bn1_name)
        X_5x5 = batchNorm(X_5x5,
                          mean=self.params[bn1_name]['m'], var=self.params[bn1_name]['v'],
                          gamma=self.params[bn1_name]['w'], beta=self.params[bn1_name]['b'],
                          numOUT=k1_shape[-1], isTrainable=self.trainable, name=bn1_name)
        X_5x5 = tf.nn.relu(X_5x5)
        
        X_5x5 = tf.pad(X_5x5, paddings=[[0, 0], [padTD[0], padTD[1]], [padLR[0], padLR[1]], [0, 0]])  #####  Zeropadding
        X_5x5 = convLayer(X_5x5, self.params[conv2_name]['w'], self.params[conv2_name]['b'], s=cnv2s,
                          isTrainable=self.trainable, name=conv2_name)
        # X_5x5 = tf.layers.batch_normalization(X_5x5, axis=1, epsilon=1e-5, name=bn2_name)
        X_5x5 = batchNorm(X_5x5,
                          mean=self.params[bn2_name]['m'], var=self.params[bn2_name]['v'],
                          gamma=self.params[bn2_name]['w'], beta=self.params[bn2_name]['b'],
                          numOUT=k2_shape[-1], isTrainable=self.trainable, name=bn2_name)
        X_5x5 = tf.nn.relu(X_5x5)
        logging.info('inception_5x5 Chain 3: shape = %s', str(X_5x5.shape))
        return X_5x5
    
    def inception_pool(self, X, cnv1s=1, padTD=(4,4), padLR=(4,4), poolSize=3, poolStride=3,
                       poolType='avg', name='3a'):
        conv_name = 'inception_%s_pool_conv'%str(name)
        bn1_name = 'inception_%s_pool_bn'%str(name)
        k1_shape = np.array(convShape[conv_name])
        k1_shape = [k1_shape[2], k1_shape[3], k1_shape[1], k1_shape[0]]
        
        # Chain 4
        if poolType=='avg':
            X_pool = tf.layers.average_pooling2d(X, pool_size=poolSize, strides=poolStride,
                                                 data_format='channels_last')
        elif poolType=='max':
            X_pool = tf.layers.max_pooling2d(X, pool_size=poolSize, strides=poolStride,
                                             data_format='channels_last')
        else:
            X_pool = X
        
        X_pool = convLayer(X_pool, self.params[conv_name]['w'], self.params[conv_name]['b'],
                           s=cnv1s, isTrainable=self.trainable, name=conv_name)
        # X_pool = tf.layers.batch_normalization(X_pool, axis=1, epsilon=1e-5, name=bn1_name)
        X_pool = batchNorm(X_pool,
                          mean=self.params[bn1_name]['m'], var=self.params[bn1_name]['v'],
                          gamma=self.params[bn1_name]['w'], beta=self.params[bn1_name]['b'],
                           numOUT=k1_shape[-1], isTrainable=self.trainable, name=bn1_name)
        X_pool = tf.nn.relu(X_pool)
        X_pool = tf.pad(X_pool, paddings=[[0, 0], [padTD[0], padTD[1]],
                                          [padLR[0], padLR[1]], [0, 0]])
        logging.info('inception_pool Chain 4: shape = %s', str(X_pool.shape))
        return X_pool
    
    def pool_pad(self, X, padTD=None, padLR=None, poolSize=3, poolStride=2,
                 poolType='avg'):
        if poolType=='avg':
            X_pool_pad = tf.layers.average_pooling2d(X, pool_size=poolSize, strides=poolStride,
                                                 data_format='channels_last')
        elif poolType=='max':
            X_pool_pad = tf.layers.max_pooling2d(X, pool_size=poolSize, strides=poolStride,
                                             data_format='channels_last')
        else:
            X_pool_pad = X

        if padTD:
            X_pool_pad = tf.pad(X_pool_pad, paddings=[[0, 0], [padTD[0], padTD[1]],
                                                      [padLR[0],padLR[1]], [0, 0]])
        logging.info('pool_pad Chain 4: shape = %s', str(X_pool_pad.shape))
        return X_pool_pad
        
            
       