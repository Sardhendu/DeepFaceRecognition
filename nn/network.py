from __future__ import division, print_function, absolute_import

from nn.model import *


def tripletLoss(predTensor, alpha=0.2):
    with tf.name_scope("TripletLoss"):
        anchor, positive, negative = predTensor[0], predTensor[1], predTensor[2]
        '''
        :param predTensor
                anchor:    The encoding of the actual image of the person
                positive:  The encodings of the image of the same person (positive)
                negative:  The encodings of the image of a different person (! the anchor person)
        :param alpha:      The penalty term that is added to the squared distance of the anchor and
                            the positive image to deliberately increase the triplet loss function so
                            that the function learns the underlying similarity much better by
                            minimizing the loss function
        :return:           LOSS
        '''
        # Mean of difference square
        positiveDist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
        negativeDist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
        
        # Calculating the loss accross all the examples in the Batch
        loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(positiveDist, negativeDist), alpha), 0))
    return loss


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
    # print tf.ty
    X = fullyConnected(X, params)
    
    return dict(inpTensor=inpTensor, output=X)

def summaryBuilder(sess, outFilePath):
    mergedSummary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(outFilePath)
    writer.add_graph(sess.graph)
    return mergedSummary, writer

    
        # with tf.Session() as test:
#     tf.set_random_seed(1)
#     y_true = (None, None, None)
#     y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
#               tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
#               tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
#     loss = tripletLoss(predTensor = y_pred)
#
#     print("loss = " + str(loss.eval()))