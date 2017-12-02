from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
from nn.Inception import convLayer, activation, Inception


def conv1(X, params):
    with tf.variable_scope('conv1'):
        X = convLayer(X, params['conv1']['w'], params['conv1']['b'], s=2,
                      isTrainable=False, name='conv1')
        X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn1')
        X = activation(X, type="relu")
        print('conv1: ', X.shape)
        
        X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        print('conv1 Zero-Padding + MAXPOOL ', X.shape)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, data_format='channels_last')
        print('conv1 Zero-Padding + MAXPOOL ', X.shape)
    
    return X


def conv2(X, params):
    with tf.variable_scope('conv2'):
        X = convLayer(X, params['conv2']['w'], params['conv2']['b'], s=1,
                      isTrainable=False, name='conv2')
        X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn2')
        X = activation(X, type="relu")
        print('conv2: ', X.shape)
        
        X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        print('conv2 Zero-Padding + MAXPOOL ', X.shape)
    
    return X


def conv3(X, params):
    with tf.variable_scope('conv3'):
        X = convLayer(X, params['conv3']['w'], params['conv3']['b'], s=1,
                      isTrainable=False, name='conv3')
        X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn3')
        X = activation(X, type="relu")
        print('conv3: ', X.shape)
        
        X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        print('conv3 Zero-Padding + MAXPOOL ', X.shape)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, data_format='channels_last')
        print('conv3 Zero-Padding + MAXPOOL ', X.shape)
    
    return X




def inception3a(X, params):
    objInception = Inception(params)
    print('Inside Inception module1: ', X.shape)
    inception3a = tf.concat(
            values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=1, padTD=(1, 1),
                                               padLR=(1, 1), name='3a'),
                    objInception.inception_5x5(X, cnv1s=1, cnv2s=1, padTD=(2, 2),
                                               padLR=(2, 2), name='3a'),
                    objInception.inception_pool(X, cnv1s=1, padTD=(3, 4), padLR=(3, 4),
                                                poolSize=3, poolStride=2, poolType='max',
                                                name='3a'),
                    objInception.inception_1x1(X, cnv1s=1, name='3a')],
            axis=-1)
    print('inception3a: ', inception3a.shape)
    return inception3a


def inception3b(X, params):
    objInception = Inception(params)
    print('Inside Inception module1: ', X.shape)
    inception3b = tf.concat(
            values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=1, padTD=(1, 1),
                                               padLR=(1, 1), name='3b'),
                    objInception.inception_5x5(X, cnv1s=1, cnv2s=1, padTD=(2, 2),
                                               padLR=(2, 2), name='3b'),
                    objInception.inception_pool(X, cnv1s=1, padTD=(4, 4), padLR=(4, 4),
                                                poolSize=3, poolStride=3, poolType='avg',
                                                name='3b'),
                    objInception.inception_1x1(X, cnv1s=1, name='3b')],
            axis=-1)
    print('inception3b: ', inception3b.shape)
    return inception3b


def inception3c(X, params):
    objInception = Inception(params)
    print('Inside Inception module1: ', X.shape)
    inception3c = tf.concat(
            values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=2, padTD=(1, 1),
                                               padLR=(1, 1), name='3c'),
                    objInception.inception_5x5(X, cnv1s=1, cnv2s=2, padTD=(2, 2),
                                               padLR=(2, 2), name='3c'),
                    objInception.pool_pad(X, padTD=(0, 1), padLR=(0, 1), poolSize=3,
                                          poolStride=2, poolType='max')],
            axis=-1)
    print('inception3c: ', inception3c.shape)
    return inception3c


def inception4a(X, params):
    print('inception_block_4a inception_block_4a inception_block_4a inception_block_4a inception_block_4a')
    objInception = Inception(params)
    print('Inside Inception module1: ', X.shape)
    inception4a = tf.concat(
            values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=1, padTD=(1, 1),
                                               padLR=(1, 1), name='4a'),
                    objInception.inception_5x5(X, cnv1s=1, cnv2s=1, padTD=(2, 2),
                                               padLR=(2, 2), name='4a'),
                    objInception.inception_pool(X, cnv1s=1, padTD=(2, 2), padLR=(2, 2),
                                                poolSize=3, poolStride=3, poolType='avg',
                                                name='4a'),
                    objInception.inception_1x1(X, cnv1s=1, name='4a')],
            axis=-1)
    print('inception4a: ', inception4a.shape)
    return inception4a


def inception4e(X, params):
    print('inception_block_4e inception_block_4e inception_block_4e inception_block_4e inception_block_4e')
    objInception = Inception(params)
    print('Inside Inception module1: ', X.shape)
    inception4e = tf.concat(
            values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=2, padTD=(1, 1),
                                               padLR=(1, 1), name='4e'),
                    objInception.inception_5x5(X, cnv1s=1, cnv2s=2, padTD=(2, 2),
                                               padLR=(2, 2), name='4e'),
                    objInception.pool_pad(X, padTD=(0, 1), padLR=(0, 1), poolSize=3,
                                          poolStride=2, poolType='max')],
            axis=-1)
    print('inception4e: ', inception4e.shape)
    return inception4e


def inception5a(X, params):
    print('inception_block_5a inception_block_5a inception_block_5a inception_block_5a inception_block_5a')
    objInception = Inception(params)
    print('Inside Inception module1: ', X.shape)
    inception5a = tf.concat(
            values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=1, padTD=(1, 1),
                                               padLR=(1, 1), name='5a'),
                    objInception.inception_pool(X, cnv1s=1, padTD=(1, 1), padLR=(1, 1),
                                                poolSize=3, poolStride=3, poolType='avg',
                                                name='5a'),
                    objInception.inception_1x1(X, cnv1s=1, name='5a')],
            axis=-1)
    print('inception5a: ', inception5a.shape)
    return inception5a


def inception5b(X, params):
    print('inception_block_5b inception_block_5b inception_block_5b inception_block_5b inception_block_5b')
    objInception = Inception(params)
    print('Inside Inception module1: ', X.shape)
    inception5b = tf.concat(
            values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=1, padTD=(1, 1),
                                               padLR=(1, 1), name='5b'),
                    objInception.inception_pool(X, cnv1s=1,padTD=(1, 1), padLR=(1, 1),
                                                poolSize=3, poolStride=2, poolType='max',
                                                name='5b'),
                    objInception.inception_1x1(X, cnv1s=1, name='5b')],
            axis=-1)
    print('inception5b: ', inception5b.shape)
    return inception5b


def getModel(imgShape, params):
    inpTensor = tf.placeholder(dtype=tf.float32, shape=[None, imgShape[0], imgShape[1], imgShape[2]])
    print('inpTensor ', inpTensor.shape)
    
    # Pad the input to make of actual size
    X = tf.pad(inpTensor, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
    X = conv1(X, params)
    X = conv2(X, params)
    X = conv3(X, params)
    X = inception3a(X, params)
    X = inception3b(X, params)
    X = inception3c(X, params)
    X = inception4a(X, params)
    X = inception4e(X, params)
    X = inception5a(X, params)
    X = inception5b(X, params)
    return dict(inpTensor=inpTensor, output=X)




