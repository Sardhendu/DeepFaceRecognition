from __future__ import division, print_function, absolute_import

from nn.network import *


def getModel(imgShape, params):
    inpTensor = tf.placeholder(dtype=tf.float32, shape=[None, imgShape[0], imgShape[1], imgShape[2]])
    logging.info('SHAPE: inpTensor %s', str(inpTensor.shape))
    
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
    X = fullyConnected(X, params)
    
    return dict(inpTensor=inpTensor, output=X)


def getModel_FT(imgShape, params):
    inpTensor = tf.placeholder(dtype=tf.float32, shape=[None, imgShape[0], imgShape[1], imgShape[2]])
    logging.info('SHAPE: inpTensor %s', str(inpTensor.shape))
    
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
    X = inception5a_FT(X)
    X = inception5b_FT(X)
    X = fullyConnected_FT(X, [736, 128])
    return X


def initNetwork(weightDict, isTrainable=False):
    logging.info('INITIALIZING THE NETWORK !! ...............................')
    if not isTrainable:
        tensorDict = getModel([96, 96, 3], params=weightDict)
    else:
        tensorDict = getModel_FT([96, 96, 3], params=weightDict)
    return tensorDict


def summaryBuilder(sess, outFilePath):
    mergedSummary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(outFilePath)
    writer.add_graph(sess.graph)
    return mergedSummary, writer