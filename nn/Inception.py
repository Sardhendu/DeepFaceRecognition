from __future__ import division, print_function, absolute_import

import tensorflow as tf


def convLayer(inpTensor, w, b, s, isTrainable, name):
    inpTensor = tf.cast(inpTensor, tf.float32)
    
    if not isTrainable:
        weights = tf.constant(w, name="weights", dtype=tf.float32)
        bias = tf.constant(b, name='biases', dtype=tf.float32)
    else:
        pass
    act = tf.nn.conv2d(inpTensor, weights, [1, s, s, 1], padding='VALID', name=name) + bias

    return act

def activation(X, type='relu'):
    if type=='relu':
        return tf.nn.relu(X)
    elif type == 'sigmoid':
        return tf.nn.sigmoid(X)

  

def conv1(X, params):
    with tf.variable_scope('conv1'):
        X = convLayer(X, params['conv1']['w'], params['conv1']['b'], s=2,
                      isTrainable=False, name='conv1')
        X = tf.layers.batch_normalization(X, axis=-1, epsilon=1e-5, name='bn1')
        X = activation(X, type="relu")
        print('conv1: ', X.shape)
        
        X = tf.pad(X, paddings=[[0,0],[1,1],[1,1],[0,0]])
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


class Inception3a():
    def __init__(self, params):
        self.params = params
        
    def inception_3a_1x1(self, X):
        # Chain 1
        conv_name = 'inception_3a_1x1_conv'
        bn_name = 'inception_3a_1x1_bn'
        X_1x1 = convLayer(X, self.params[conv_name]['w'], self.params[conv_name]['b'], s=1,
                          isTrainable=False, name=conv_name)
        X_1x1 = tf.layers.batch_normalization(X_1x1, axis=1, epsilon=1e-5, name=bn_name)
        X_1x1 = tf.nn.relu(X_1x1)
        print('Chain 1: ', X_1x1.shape)
        return X_1x1
        
    def inception_3a_3x3(self, X):
        # Chain 2
        conv1_name = 'inception_3a_3x3_conv1'
        bn1_name = 'inception_3a_3x3_bn1'
        conv2_name = 'inception_3a_3x3_conv2'
        bn2_name = 'inception_3a_3x3_bn2'
        
        X_3x3 = convLayer(X, self.params[conv1_name]['w'], self.params[conv1_name]['b'], s=1,
                          isTrainable=False, name=conv1_name)
        X_3x3 = tf.layers.batch_normalization(X_3x3, axis=1, epsilon=1e-5, name=bn1_name)
        X_3x3 = tf.nn.relu(X_3x3)
        
        X_3x3 = tf.pad(X_3x3, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])  #####  Zeropadding
        X_3x3 = convLayer(X_3x3, self.params[conv2_name]['w'], self.params[conv2_name]['b'], s=1,
                          isTrainable = False, name = conv2_name)
        X_3x3 = tf.layers.batch_normalization(X_3x3, axis=1, epsilon=1e-5, name=bn2_name)
        X_3x3 = tf.nn.relu(X_3x3)
        print('Chain 2: ', X_3x3.shape)
        return X_3x3

    def inception_3a_5x5(self, X):
        # Chain 3
        conv1_name='inception_3a_5x5_conv1'
        bn1_name='inception_3a_5x5_bn1'
        conv2_name='inception_3a_5x5_conv2'
        bn2_name='inception_3a_5x5_bn2'
        
        X_5x5 = convLayer(X, self.params[conv1_name]['w'], self.params[conv1_name]['b'], s=1,
                          isTrainable=False, name=conv1_name)
        X_5x5 = tf.layers.batch_normalization(X_5x5, axis=1, epsilon=1e-5, name=bn1_name)
        X_5x5 = tf.nn.relu(X_5x5)
        
        X_5x5 = tf.pad(X_5x5, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]])  #####  Zeropadding
        X_5x5 = convLayer(X_5x5, self.params[conv2_name]['w'], self.params[conv2_name]['b'], s=1,
                          isTrainable=False, name=conv2_name)
        X_5x5 = tf.layers.batch_normalization(X_5x5, axis=1, epsilon=1e-5, name=bn2_name)
        X_5x5 = tf.nn.relu(X_5x5)
        print('Chain 3: ', X_5x5.shape)
        return X_5x5
        
    def inception_3a_maxpool(self, X):
        conv_name = 'inception_3a_pool_conv'
        bn1_name='inception_3a_pool_bn'
        
        # Chain 4
        X_pool = tf.layers.max_pooling2d(X, pool_size=3, strides=2, data_format='channels_last')
        X_pool = convLayer(X_pool, self.params[conv_name]['w'], self.params[conv_name]['b'], s=1,
                          isTrainable=False, name=conv_name)
        X_pool = tf.layers.batch_normalization(X_pool, axis=1, epsilon=1e-5, name=bn1_name)
        X_pool = tf.nn.relu(X_pool)
        X_pool = tf.pad(X_pool, paddings=[[0, 0], [3, 4], [3, 4], [0, 0]])
        print('Chain 4: ', X_pool.shape)
        
        return X_pool
    
    def inception3a(self, X):
        print ('Inside Inception module1: ', X.shape)
        inception3a = tf.concat(values=[self.inception_3a_3x3(X),
                                        self.inception_3a_5x5(X),
                                        self.inception_3a_maxpool(X),
                                        self.inception_3a_1x1(X)], axis=-1)
        print('inception3a: ', inception3a.shape)
        return inception3a


class Inception3b():
    def __init__(self, params):
        self.params = params
        
    def inception_3b_1x1(self, X):
        conv_name = 'inception_3b_1x1_conv'
        bn_name = 'inception_3b_1x1_bn'
        X_1x1 = convLayer(X, self.params[conv_name]['w'], self.params[conv_name]['b'], s=1,
                          isTrainable=False, name=conv_name)
        X_1x1 = tf.layers.batch_normalization(X_1x1, axis=1, epsilon=1e-5, name=bn_name)
        X_1x1 = tf.nn.relu(X_1x1)
        print('Chain 1: ', X_1x1.shape)
        return X_1x1

    def inception_3b_3x3(self, X):
        # Chain 2
        conv1_name = 'inception_3b_3x3_conv1'
        bn1_name = 'inception_3b_3x3_bn1'
        conv2_name = 'inception_3b_3x3_conv2'
        bn2_name = 'inception_3b_3x3_bn2'
    
        X_3x3 = convLayer(X, self.params[conv1_name]['w'], self.params[conv1_name]['b'], s=1,
                          isTrainable=False, name=conv1_name)
        X_3x3 = tf.layers.batch_normalization(X_3x3, axis=1, epsilon=1e-5, name=bn1_name)
        X_3x3 = tf.nn.relu(X_3x3)
        
        X_3x3 = tf.pad(X_3x3, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])  #####  Zeropadding
        X_3x3 = convLayer(X_3x3, self.params[conv2_name]['w'], self.params[conv2_name]['b'], s=1,
                          isTrainable=False, name=conv2_name)
        X_3x3 = tf.layers.batch_normalization(X_3x3, axis=1, epsilon=1e-5, name=bn2_name)
        X_3x3 = tf.nn.relu(X_3x3)
        print('Chain 2: ', X_3x3.shape)
        return X_3x3

    def inception_3b_5x5(self, X):
        # Chain 3
        conv1_name = 'inception_3b_5x5_conv1'
        bn1_name = 'inception_3b_5x5_bn1'
        conv2_name = 'inception_3b_5x5_conv2'
        bn2_name = 'inception_3b_5x5_bn2'
    
        X_5x5 = convLayer(X, self.params[conv1_name]['w'], self.params[conv1_name]['b'], s=1,
                          isTrainable=False, name=conv1_name)
        X_5x5 = tf.layers.batch_normalization(X_5x5, axis=1, epsilon=1e-5, name=bn1_name)
        X_5x5 = tf.nn.relu(X_5x5)
    
        X_5x5 = tf.pad(X_5x5, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]])  #####  Zeropadding
        X_5x5 = convLayer(X_5x5, self.params[conv2_name]['w'], self.params[conv2_name]['b'], s=1,
                          isTrainable=False, name=conv2_name)
        X_5x5 = tf.layers.batch_normalization(X_5x5, axis=1, epsilon=1e-5, name=bn2_name)
        X_5x5 = tf.nn.relu(X_5x5)
        print('Chain 3: ', X_5x5.shape)
        return X_5x5

    def inception_3b_maxpool(self, X):
        conv_name = 'inception_3b_pool_conv'
        bn1_name = 'inception_3b_pool_bn'
    
        # Chain 4
        X_pool = tf.layers.average_pooling2d(X, pool_size=3, strides=3, data_format='channels_last')
        X_pool = convLayer(X_pool, self.params[conv_name]['w'], self.params[conv_name]['b'], s=1,
                           isTrainable=False, name=conv_name)
        X_pool = tf.layers.batch_normalization(X_pool, axis=1, epsilon=1e-5, name=bn1_name)
        X_pool = tf.nn.relu(X_pool)
        X_pool = tf.pad(X_pool, paddings=[[0, 0], [4, 4], [4, 4], [0, 0]])
        print('Chain 4: ', X_pool.shape)
        return X_pool

    def inception3b(self, X):
        print('Inside Inception module1: ', X.shape)
        inception3b = tf.concat(values=[self.inception_3b_3x3(X),
                                        self.inception_3b_5x5(X),
                                        self.inception_3b_maxpool(X),
                                        self.inception_3b_1x1(X)], axis=-1)
        print('inception3b: ', inception3b.shape)
        return inception3b
    
    
    
def getModel(imgShape, params):
    inpTensor = tf.placeholder(dtype=tf.float32, shape=[None, imgShape[0], imgShape[1], imgShape[2]])
    print('inpTensor ', inpTensor.shape)
    
    # Pad the input to make of actual size
    X = tf.pad(inpTensor, paddings=[[0,0],[3, 3], [3, 3], [0,0]])
    X = conv1(X, params)
    X = conv2(X, params)
    X = conv3(X, params)
    X = Inception3a(params).inception3a(X)
    X = Inception3b(params).inception3b(X)
    
    return dict(inpTensor=inpTensor, output=X)
    