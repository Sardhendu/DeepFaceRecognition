from __future__ import division, print_function, absolute_import
import numpy as np

import tensorflow  as tf


### Tensor flow implementation
'''
    Thwe model follows the nn4_small2_c1 implementation of Face net model
'''


# https://github.com/davidsandberg/facenet/blob/master/tmp/nn4_small2_v1.py


def inception_3a(X):
    '''
    :param inp: Input image
    :return:
    '''
    '''
        This module can be thought of one Inception block
    '''
    
    # Chain 1
    X_1x1 = tf.layers.conv2d(X, filters=64, kernel_size=(1, 1), data_format='channels_last',
                             name='inception_3a_1x1_conv')
    X_1x1 = tf.layers.batch_normalization(X_1x1, axis=1, epsilon=1e-5, name='inception_3a_1x1_bn')
    X_1x1 = tf.nn.relu(X_1x1)
    print('Chain 1: ', X_1x1.shape)
    
    # Chain 2
    X_3x3 = tf.layers.conv2d(X, filters=96, kernel_size=(1, 1), data_format='channels_last',
                             name='inception_3a_3x3_conv1')
    X_3x3 = tf.layers.batch_normalization(X_3x3, axis=1, epsilon=1e-5, name='inception_3a_3x3_bn1')
    X_3x3 = tf.nn.relu(X_3x3)
    X_3x3 = tf.pad(X_3x3, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])  #####  Zeropadding
    X_3x3 = tf.layers.conv2d(X_3x3, filters=128, kernel_size=(3, 3), data_format='channels_last',
                             name='inception_3a_3x3_conv2')
    X_3x3 = tf.layers.batch_normalization(X_3x3, axis=1, epsilon=1e-5, name='inception_3a_3x3_bn2')
    X_3x3 = tf.nn.relu(X_3x3)
    print('Chain 2: ', X_3x3.shape)
    
    # Chain 2
    X_5x5 = tf.layers.conv2d(X, filters=16, kernel_size=(1, 1), data_format='channels_last',
                             name='inception_3a_5x5_conv1')
    X_5x5 = tf.layers.batch_normalization(X_5x5, axis=1, epsilon=1e-5, name='inception_3a_5x5_bn1')
    X_5x5 = tf.nn.relu(X)
    X_5x5 = tf.pad(X_5x5, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]])  #####  Zeropadding
    X_5x5 = tf.layers.conv2d(X_5x5, filters=32, kernel_size=(5, 5), data_format='channels_last',
                             name='inception_3a_5x5_conv2')
    X_5x5 = tf.layers.batch_normalization(X_5x5, axis=1, epsilon=1e-5, name='inception_3a_5x5_bn2')
    X_5x5 = tf.nn.relu(X_5x5)
    print('Chain 3: ', X_5x5.shape)
    
    # Chain 4
    X_pool = tf.layers.max_pooling2d(X, pool_size=3, strides=2, data_format='channels_last')
    X_pool = tf.layers.conv2d(X_pool, filters=32, kernel_size=(1, 1), data_format='channels_last',
                              name='inception_3a_pool_conv')
    X_pool = tf.layers.batch_normalization(X_pool, axis=1, epsilon=1e-5, name='inception_3a_pool_bn')
    X_pool = tf.nn.relu(X_pool)
    X_pool = tf.pad(X_pool, paddings=[[0, 0], [3, 4], [3, 4], [0, 0]])
    print('Chain 3: ', X_5x5.shape)
    
    inception3a = tf.concat(values=[X_3x3, X_5x5, X_pool, X_1x1], axis=-1)
    
    print('inception3a: ', inception3a.shape)
    
    return inception3a


def howMuchToPad(imgSize, filterSize, padSize, strides):
    fm_nH = (imgSize[0] + 2 * padSize[0] - filterSize[0]) / strides[0] + 1
    fm_nW = (imgSize[1] + 2 * padSize[1] - filterSize[1]) / strides[1] + 1
    print('Image Size with the given pad ', [fm_nH, fm_nW])
    print('You should pad with %s both every boundary ', (np.array(imgSize) - np.array([fm_nH, fm_nW])) / 2)
    return [fm_nH, fm_nW]


print(howMuchToPad(imgSize=[96, 96], filterSize=[7, 7], padSize=[0, 0], strides=[1, 1]))


def conv1(X):
    # Layer 1
    X = tf.layers.conv2d(X, filters=64, kernel_size=(7, 7), strides=2, padding='SAME', data_format="channels_last",
                         name="conv1")
    X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn1')
    X = tf.nn.relu(X)
    print('conv1: ', X.shape)
    
    # Zero-Padding + MAXPOOL
    X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    print('conv1 Zero-Padding + MAXPOOL ', X.shape)
    X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, data_format='channels_last')
    print('conv1 Zero-Padding + MAXPOOL ', X.shape)
    
    return X


def conv2(X):
    # Layer 2:
    X = tf.layers.conv2d(X, filters=64, kernel_size=(1, 1), strides=1, data_format="channels_last", name="conv2")
    X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn2')
    X = tf.nn.relu(X)
    print('conv2: ', X.shape)
    
    X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    print('conv1 Zero-Padding + MAXPOOL ', X.shape)
    
    return X


def conv3(X):
    # Layer 1
    X = tf.layers.conv2d(X, filters=192, kernel_size=(3, 3), strides=1, data_format="channels_last", name="conv3")
    X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn3')
    X = tf.nn.relu(X)
    print('conv1: ', X.shape)
    
    # Zero-Padding + MAXPOOL
    X = tf.pad(X, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    print('conv1 Zero-Padding + MAXPOOL ', X.shape)
    X = tf.layers.max_pooling2d(X, pool_size=3, strides=2, data_format='channels_last')
    print('conv1 Zero-Padding + MAXPOOL ', X.shape)
    
    return X


def getModel(inputShape):
    # Declare and innput placeholder to assign the input in the run time
    inpTensor = tf.placeholder(dtype=tf.float32, shape=[None, inputShape[0], inputShape[1], inputShape[2]])
    print('inpTensor ', inpTensor.shape)
    
    # Pad the input to make of actual size
    # paddings = tf.constant([[0,0],[3, 3], [3, 3], [0,0]])
    # X = tf.pad(inpTensor, paddings=paddings)
    # print ('paddings: ', X.shape)
    
    X = conv1(inpTensor)
    X = conv2(X)
    X = conv3(X)
    X = inception_3a(X)
    
    # tf.assign.
    
    # inpTensor(?, 96, 96, 3)
    # paddings: (?, 102, 102, 3)
    # conv1: (?, 48, 48, 64)
    # conv1
    # Zero - Padding + MAXPOOL(?, 50, 50, 64)
    # conv1
    # Zero - Padding + MAXPOOL(?, 24, 24, 64)
    # conv2: (?, 24, 24, 64)
    # conv1
    # Zero - Padding + MAXPOOL(?, 26, 26, 64)
    # conv1: (?, 24, 24, 192)
    # conv1
    # Zero - Padding + MAXPOOL(?, 26, 26, 192)
    # conv1
    # Zero - Padding + MAXPOOL(?, 12, 12, 192)
    
    
    
    
    
    
    # # Layer 1:
    # tf.layers.conv2d(imgND, filters)
    #


FRmodel = getModel(inputShape=[96, 96, 3])



# tf.nn.conv2d(xIN, w, [1, stride, stride, 1], padding=padding) + b


# weightsNameList = ['inception_3a_1x1_conv', 'inception_3a_1x1_bn',
#                    'inception_3a_3x3_conv1', 'inception_3a_3x3_bn1',
#                    'inception_3a_3x3_conv2', 'inception_3a_3x3_bn2',
#                    'inception_3a_5x5_conv1', 'inception_3a_5x5_bn1',
#                    'inception_3a_5x5_conv2', 'inception_3a_5x5_bn2',
#                    'inception_3a_pool_conv', 'inception_3a_pool_bn']



def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_first', name=layer+'_conv'+num)(x)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding, data_format='channels_first')(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=layer+'_conv'+'2')(tensor)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor