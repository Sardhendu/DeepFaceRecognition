from __future__ import division, print_function, absolute_import

from nn.network import *


def trainModel_FT(imgShape, params, init_wght_type='pretrained'):
    inpTensor = tf.placeholder(dtype=tf.float32, shape=[None, imgShape[0], imgShape[1], imgShape[2]])
    logging.info('SHAPE: inpTensor %s', str(inpTensor.shape))
    
    # Pad the input to make of actual size
    X = tf.pad(inpTensor, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
    X = conv1(X, params)
    X = conv2(X, params)
    X = conv3(X, params)
    X = inception3a(X, params, trainable=False)
    X = inception3b(X, params, trainable=False)
    X = inception3c(X, params, trainable=False)
    X = inception4a(X, params, trainable=False)
    X = inception4e(X, params, trainable=False)
    if init_wght_type=='pretrained':
        logging.info('Initializing the last layer weights with inception pre-trained weight but the parameter is '
                     'trainable')
        X = inception5a(X, params, trainable=True)
        X = inception5b(X, params, trainable=True)
        X = fullyConnected(X, params, trainable=True)
    elif init_wght_type=='random':
        logging.info('Initializing the last layer weights with random values and the parameter is trainable')
        X = inception5a_FT(X)
        X = inception5b_FT(X)
        X = fullyConnected_FT(X, [736, 128])
    else:
        raise ValueError('Provide a valid weight initialization type')
    return dict(inpTensor=inpTensor, output=X)


def getEmbeddings(imgShape, params, which_model='inception'):
    inpTensor = tf.placeholder(dtype=tf.float32, shape=[None, imgShape[0], imgShape[1], imgShape[2]])
    logging.info('SHAPE: inpTensor %s', str(inpTensor.shape))
    
    # Pad the input to make of actual size
    X = tf.pad(inpTensor, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
    X = conv1(X, params)
    X = conv2(X, params)
    X = conv3(X, params)
    X = inception3a(X, params, trainable=False)
    X = inception3b(X, params, trainable=False)
    X = inception3c(X, params, trainable=False)
    X = inception4a(X, params, trainable=False)
    X = inception4e(X, params, trainable=False)
    if which_model == 'inception':
        X = inception5a(X, params, trainable=False)
        X = inception5b(X, params, trainable=False)
        X = fullyConnected(X, params, trainable=False)
    elif which_model == 'finetune':
        X = inception5a(X, params, trainable=False)
        X = inception5b(X, params, trainable=False)
        X = fullyConnected(X, params, trainable=False)
    else:
        raise ValueError('Model name not recognized')
    
    return dict(inpTensor=inpTensor, output=X)

def initNetwork(weightDict, init_wght_type, isTrainable=False):
    logging.info('INITIALIZING THE NETWORK !! ...............................')
    if not isTrainable:
        encodingDict = getModel([96, 96, 3], params=weightDict)
    else:
        img_per_label = 6
        num_labels = 3
        alpha = 0.01
        learning_rate = 0.0001
        encodingDict = getModel_FT([96, 96, 3], params=weightDict, init_wght_type=init_wght_type)
        encodingDict = loss(encodingDict, img_per_label, num_labels, alpha)
        encodingDict = optimize(encodingDict, learning_rate)
    return encodingDict

def summaryBuilder(sess, outFilePath):
    mergedSummary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(outFilePath)
    writer.add_graph(sess.graph)
    return mergedSummary, writer