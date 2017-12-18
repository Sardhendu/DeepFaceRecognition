from __future__ import division, print_function, absolute_import
import tensorflow as tf
from nn.inception import convLayer, activation, Inception, batchNorm
from nn.inception_finetune import Inception_FT


def conv1(X, params):
    conv_name = 'conv1'
    bn_name = 'bn1'
    with tf.variable_scope(conv_name):
        X = convLayer(X, params[conv_name]['w'], params[conv_name]['b'], s=2,
                      isTrainable=False, name=conv_name)
        # X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn1')
        X = batchNorm(X,
                      mean=params[bn_name]['m'], var=params[bn_name]['v'],
                      gamma=params[bn_name]['w'], beta=params[bn_name]['b'],
                      isTrainable=False, name=bn_name)
        X = activation(X, type="relu")
        print('conv1: ', X.shape)
        
        X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        print('conv1 Zero-Padding + MAXPOOL ', X.shape)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, data_format='channels_last')
        print('conv1 Zero-Padding + MAXPOOL ', X.shape)
    
    return X


def conv2(X, params):
    conv_name = 'conv2'
    bn_name = 'bn2'
    with tf.variable_scope(conv_name):
        X = convLayer(X, params[conv_name]['w'], params[conv_name]['b'], s=1,
                      isTrainable=False, name=conv_name)
        # X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn2')
        X = batchNorm(X,
                      mean=params[bn_name]['m'], var=params[bn_name]['v'],
                      gamma=params[bn_name]['w'], beta=params[bn_name]['b'],
                      isTrainable=False, name=bn_name)
        X = activation(X, type="relu")
        print('conv2: ', X.shape)
        
        X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        print('conv2 Zero-Padding + MAXPOOL ', X.shape)
    
    return X


def conv3(X, params):
    conv_name = 'conv3'
    bn_name = 'bn3'
    with tf.variable_scope(conv_name):
        X = convLayer(X, params[conv_name]['w'], params[conv_name]['b'], s=1,
                      isTrainable=False, name=conv_name)
        # X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn3')
        X = batchNorm(X,
                      mean=params[bn_name]['m'], var=params[bn_name]['v'],
                      gamma=params[bn_name]['w'], beta=params[bn_name]['b'],
                      isTrainable=False, name=bn_name)
        X = activation(X, type="relu")
        print('conv3: ', X.shape)
        
        X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        print('conv3 Zero-Padding + MAXPOOL ', X.shape)
        X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, data_format='channels_last')
        print('conv3 Zero-Padding + MAXPOOL ', X.shape)
    
    return X




def inception3a(X, params):
    with tf.name_scope("Inception3a"):
        objInception = Inception(params)
        print('Inside Inception module 3a: ', X.shape)
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
    with tf.name_scope("Inception3b"):
        objInception = Inception(params)
        print('Inside Inception module 3b: ', X.shape)
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
    with tf.name_scope("Inception3c"):
        objInception = Inception(params)
        print('Inside Inception module 3c: ', X.shape)
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
    with tf.name_scope("Inception4a"):
        objInception = Inception(params)
        print('Inside Inception module 4a: ', X.shape)
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
    with tf.name_scope("Inception4e"):
        objInception = Inception(params)
        print('Inside Inception module 4e: ', X.shape)
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
    with tf.name_scope("Inception5a"):
        objInception = Inception(params)
        print('Inside Inception module 5a: ', X.shape)
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
    with tf.name_scope("Inception5b"):
        objInception = Inception(params)
        print('Inside Inception module 5b: ', X.shape)
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



def fullyConnected(X, params):
    with tf.name_scope("InceptionFC"):
        X = tf.cast(X, tf.float32)
        w = tf.constant(params['dense']['w'], name="weights", dtype=tf.float32)
        b = tf.constant(params['dense']['b'], name="bias", dtype=tf.float32)
        X = tf.layers.average_pooling2d(X, pool_size=3, strides=1,
                                        data_format='channels_last')
        print ('X after FC pool: ', X.shape)
        X = tf.contrib.layers.flatten(X)
        print('X after X Flattened: ', X.shape)
        # print ('sdcdsddf ', params['dense']['w'].dtype, params['dense']['b'].dtype)
        X = tf.add(tf.matmul(X, w), b)
        print('X after FC Matmul: ', X.shape)

        # The output encoding identifies [batchSize, 128], L2 norm is perform for each
        # training example (per record), Formula: a / pow(max(sum(a**2), 1e-5), 0.5)
        X = tf.nn.l2_normalize(X, dim=1, epsilon=1e-12, name='L2_norm')
    return X



########################################################################################
## FINE TUNE LAST FEW LAYERS: ONLY TRAIN THE LAST FEW LAYERS FOR YOUR IMAGES
########################################################################################

def inception5a_FT(X):
    with tf.name_scope("Inception5a_FT"):
        print('Inside Inception module 5a FT: ', X.shape)
        objInception = Inception_FT()
        inception5a = tf.concat(
                values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=1,
                                                   padTD=(1, 1), padLR=(1, 1), name='5a'),
                        objInception.inception_pool(X, cnv1s=1, padTD=(1, 1), padLR=(1, 1),
                                                    poolSize=3, poolStride=3, poolType='avg',
                                                    name='5a'),
                        objInception.inception_1x1(X, cnv1s=1, name='5a')],
                axis=-1)
        print('inception5a: ', inception5a.shape)
    return inception5a


def inception5b_FT(X):
    with tf.name_scope("Inception5a_FT"):
        print('Inside Inception module 5a FT: ', X.shape)
        objInception = Inception_FT()
        inception5a = tf.concat(
                values=[objInception.inception_3x3(X, cnv1s=1, cnv2s=1, padTD=(1, 1),
                                                   padLR=(1, 1), name='5b'),
                        objInception.inception_pool(X, cnv1s=1, padTD=(1, 1), padLR=(1, 1),
                                                    poolSize=3, poolStride=2, poolType='max',
                                                    name='5b'),
                        objInception.inception_1x1(X, cnv1s=1, name='5b')],
                axis=-1)
        print('inception5a: ', inception5a.shape)
    return inception5a


def fullyConnected_FT(X, k_shape):
    name = "InceptionFC_FT"
    with tf.name_scope(name):
        X = tf.cast(X, tf.float32)
        X = tf.layers.average_pooling2d(X, pool_size=3, strides=1,
                                        data_format='channels_last')
        print('X after FC pool: ', X.shape)
        
        with tf.variable_scope(name+'_wb'):
            w = tf.get_variable(
                    dtype='float32',
                    shape=k_shape,
                    initializer=tf.truncated_normal_initializer(
                            stddev=0.1, seed=6752
                    ),
                    name="convWeight",
                    trainable=True
            )
            
            b = tf.get_variable(
                    dtype='float32',
                    shape=[k_shape[-1]],
                    initializer=tf.constant_initializer(1),
                    name="convBias",
                    trainable=True
            
            )

        tf.summary.histogram("FC_Weights", w)
        tf.summary.histogram("FC_bias", b)
        # Flatten the pooled output (cnvrt [batchSize, 1, 1, 736] to [batchSize, 736])
        X = tf.contrib.layers.flatten(X)
        print('X after X Flattened: ', X.shape)
        # print ('sdcdsddf ', params['dense']['w'].dtype, params['dense']['b'].dtype)
        X = tf.add(tf.matmul(X, w), b)
        print('X after FC Matmul: ', X.shape)
        
        # The output encoding identifies [batchSize, 128], L2 norm is perform for each
        # training example (per record), Formula: a / pow(max(sum(a**2), 1e-5), 0.5)
        # Basically we normalize the embedding output per image
        X = tf.nn.l2_normalize(X, dim=1, epsilon=1e-12, name='L2_norm')
    return X
#
# import numpy as np
# X = np.random.rand(1,3,3,1024)
# X = tf.cast(X, dtype=tf.float32)
# X = inception5a_FT(X)
# X = inception5b_FT(X)
# X = fullyConnected_FT(X, k_shape=[736, 128])