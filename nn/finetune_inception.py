from __future__ import division, print_function, absolute_import

import tensorflow as tf
from nn.load_params import convShape
import numpy as np

def convLayer_FT(inpTensor, kShape, s, name):
    inpTensor = tf.cast(inpTensor, tf.float32)
    
    with tf.variable_scope(name+'_wb'):
        w = tf.get_variable(
                dtype='float32',
                shape=kShape,
                initializer=tf.truncated_normal_initializer(
                         stddev=0.1, seed=6752
                ),
                name="convWeight"
        )
    
        b = tf.get_variable(
                dtype='float32',
                shape=[kShape[-1]],
                initializer=tf.constant_initializer(1),
                name="convBias"
    
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

    def inception1x1(self, X, name='5a'):
        conv_name = 'inception_%s_1x1_conv' % str(name)
        bn_name = 'inception_%s_1x1_bn' % str(name)
        kShape = np.array(convShape[conv_name])
        kShape = [kShape[2], kShape[3], kShape[1], kShape[0]]
        X = convLayer_FT(X, kShape, s=1, name=conv_name)
        print (X.get_shape().as_list())
        X = batchNorm_FT(X, kShape[-1], axis=[0,1,2], name=bn_name)
        print(X.get_shape().as_list())


X = np.random.rand(1,3,3,1024)
X = tf.cast(X, dtype=tf.float32)
Inception_FT().inception1x1(X)