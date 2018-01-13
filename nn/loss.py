import tensorflow as tf
import numpy as np
from itertools import combinations
import logging
import config

from config import myNet



def tripletLoss(anchor, positive, negative, alpha=0.2):
    # = predTensor[0], predTensor[1], predTensor[2]
    '''
    :param anchor:      array of encodings of several actual image of the person
    :param positive:    array of encodings of several hard actual image of the person
    :param negative:    array of encodings of several hard actual image of other people
    :param alpha:       The penalty term that is added to the squared distance of the anchor and
                     the positive image to deliberately increase the triplet loss function so
                     that the function learns the underlying similarity much better by
                     minimizing the loss function
    :return:            LOSS
    '''
    
    # Mean of difference square
    positiveDist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    negativeDist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    
    # Calculating the loss accross all the examples in the Batch
    triplet_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(positiveDist, negativeDist), alpha), 0))
    return triplet_loss


def getTriplets(batch_embedding, img_per_label, num_labels, alpha):
    '''
        MODULE COMPLETE:         Numpy implementation of triplet selection
        :param batch_embedding:  The Input batch encoding (embeddings) [batchSize, 128]
        :param img_per_label:    Number of images per label
        :param num_labels:       Number of labels (classes)
        :param alpha:            The penalty for hard negative and hard positive
        :return:                 The triplet indexes
    '''
    batch_tripet_idx = []
    idx_arr = np.arange(len(batch_embedding))
    
    ##################  TO BE REMOVE
    sseedd_arr = []
    ##################  REMOVE
    
    for i in np.arange(num_labels):
        pos_idxs = np.arange(i * img_per_label, i * img_per_label + img_per_label)
        neg_idxs = np.setdiff1d(idx_arr, pos_idxs)
        
        compare_point = -1  # used to avoid redundancy in calculating SSE between anchor and all negative
        # Get all combination of Anchor and positive
        for anc_idx, pos_idx in combinations(pos_idxs, 2):
            # print (anc_idx, pos_idx)
            if anc_idx != compare_point:
                compare_point += 1
                # Get the sum of squared distance between anchor and negative
                anc_VS_neg_ndarr = np.sum(np.square(
                        batch_embedding[anc_idx] - batch_embedding[neg_idxs]), 1
                )
            # print(anc_VS_neg_ndarr.shape)
            # Get the sum of squared distance between anchor and positive
            anc_VS_pos = np.sum(
                    np.square(batch_embedding[anc_idx] - batch_embedding[pos_idx]))
            # print (anc_VS_pos)
            # print (anc_VS_neg_ndarr)
            hard_neg_idx = np.where(anc_VS_neg_ndarr - anc_VS_pos < alpha)[0]
            # print (hard_neg_idx)
            # print('')
            # Randomly sample 1 record from the hard negative idx, and create a triplet
            if len(hard_neg_idx) > 0:
                
                if config.triplet_seed_idx == len(config.seed_arr) - 1:
                    config.triplet_seed_idx = 0
                sseedd_arr.append(config.seed_arr[config.triplet_seed_idx])
                np.random.seed(config.seed_arr[config.triplet_seed_idx])
                
                # logging.info('Shuffling hard negative selection with seed idx = %s and seed %s',
                # str(config.triplet_seed_idx), str(config.seed_arr[config.triplet_seed_idx]))
                config.triplet_seed_idx += 1
                
                np.random.shuffle(hard_neg_idx)
                # logging.info('Hard Negative Index %s', str(hard_neg_idx))
                
                np.random.shuffle(hard_neg_idx)
                
                # logging.info('Shuffled Hard Negative Index %s', str(hard_neg_idx))
                rnd_idx = hard_neg_idx[0]
                
                # logging.info('chosen Hard Negative Index %s', str([anc_idx, pos_idx, neg_idxs[rnd_idx]]))
                # Get triplet indexes in order to work on offline/online mode
                batch_tripet_idx.append([anc_idx, pos_idx, neg_idxs[rnd_idx]])
            # else:
            #     logging.info('No hard negative index found for anc_idx = %s and  pos_idx = %s ', str(anc_idx),
            # str(pos_idx))
    logging.info('SEED USED %s', str(sseedd_arr))
    logging.info('TRIPLETS %s == %s', str(len(batch_tripet_idx)), str(batch_tripet_idx))
    # print ()
    return [batch_tripet_idx]



########################################################################################
## MAIN CALL: TRIPLET LOSS AND OPTIMIZATION
########################################################################################

def loss(embeddings):
    '''
        Implementing the Tensor Graph for Triplet loss function.
        The output tripletIDX is a numpy ND array.
    '''
    tripletIDX = tf.py_func(getTriplets,
                            [embeddings, myNet['img_per_label'],
                             myNet['num_labels'],
                             myNet['triplet_selection_alpha']],
                             tf.int64)
    with tf.name_scope('triplet_loss'):
        triplet_loss = tripletLoss(
            tf.gather(tf.cast(embeddings, dtype=tf.float32), tripletIDX[:,0]),
            tf.gather(tf.cast(embeddings, dtype=tf.float32), tripletIDX[:,1]),
            tf.gather(tf.cast(embeddings, dtype=tf.float32), tripletIDX[:,2]),
            alpha=myNet['triplet_loss_penalty'])
    tf.summary.scalar('triplet_loss', triplet_loss)
    # embeddingDict['loss'] = loss
    return triplet_loss





debugg = False
if debugg:
    np.random.seed(327)
    batch_embedding = np.random.rand(40, 2)
    batch_size, num_embeddings = batch_embedding.shape
    img_per_label = 10
    num_labels = int( batch_size /img_per_label)

    alpha = 0.01


    batch_tripet_idx = getTriplets(batch_embedding, img_per_label, num_labels, 0.2)

    print (batch_tripet_idx)
