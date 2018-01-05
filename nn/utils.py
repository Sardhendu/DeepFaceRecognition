import tensorflow as tf
import numpy as np
from itertools import combinations
import logging
import config

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
   
    with tf.name_scope("TripletLoss"):
        # Mean of difference square
        positiveDist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        negativeDist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        # Calculating the loss accross all the examples in the Batch
        loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(positiveDist, negativeDist), alpha), 0))
    return loss


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
    for i in np.arange(num_labels):
        pos_idxs = np.arange(i * img_per_label, i * img_per_label + img_per_label)
        neg_idxs = np.setdiff1d(idx_arr, pos_idxs)

        compare_point = -1  # used to avoid redundancy in calculating SSE between anchor and all negative
        # Get all combination of Anchor and positive
        for anc_idx, pos_idx in combinations(pos_idxs, 2):
            if anc_idx != compare_point:
                compare_point += 1
                # Get the sum of squared distance between anchor and positive
                anc_VS_neg_ndarr = np.sum(np.square(
                        batch_embedding[anc_idx] - batch_embedding[neg_idxs]), 1
                )
            
            # Get the sum of squared distance between anchor and positive
            anc_VS_pos = np.sum(
                    np.square(batch_embedding[anc_idx] - batch_embedding[pos_idx]))
            
            hard_neg_idx = np.where(anc_VS_neg_ndarr - anc_VS_pos < alpha)[0]
            
            # Randomly sample 1 record from the hard negative idx, and create a triplet
            if len(hard_neg_idx) > 0:
                if config.triplet_seed_idx == len(config.seed_arr) - 1:
                    config.triplet_seed_idx = 0
                np.random.seed(config.seed_arr[config.triplet_seed_idx])
                
                # logging.info('Shuffling hard negative selection with seed idx = %s and seed %s', str(config.triplet_seed_idx), str(config.seed_arr[config.triplet_seed_idx]))
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
            #     logging.info('No hard negative index found for anc_idx = %s and  pos_idx = %s ', str(anc_idx), str(pos_idx))
    logging.info('TRIPLETS %s == %s', str(len(batch_tripet_idx)), str(batch_tripet_idx))
    return [batch_tripet_idx]


#
# debugg = True
# if debugg:
#     np.random.seed(327)
#     batch_embedding = np.random.rand(40, 2)
#     batch_size, num_embeddings = batch_embedding.shape
#     img_per_label = 10
#     num_labels = int( batch_size /img_per_label)
#
#     alpha = 0.01
#
#
#     batch_tripet_idx = getTriplets(batch_embedding, img_per_label, num_labels, 0.2)
#
#     print (batch_tripet_idx)



def getTriplets_TF(batch_embedding, img_per_label, num_labels, alpha=0.01):
    '''
    MODULE NOT COMPLETE:     Tensor flow implementation of Triplet selection
    :param batch_embedding:  The Input batch encoding (embeddings) [batchSize, 128]
    :param img_per_label:    Number of images per label
    :param num_labels:       Number of labels (classes)
    :param alpha:            The penalty for hard negative and hard positive
    :return:                 The triplet indexes
    '''
    batch_tripet_idxs = tf.constant([])
    # print ('yywyruiwyriuwyuiryuwryuwuier ', batch_tripet_idxs.eval())
    idx_arr = np.arange(batch_embedding.get_shape().as_list()[0])
    for i in np.arange(num_labels):
        # print ('2423423412425453463423423425345')
        pos_idxs = np.arange(i * img_per_label, i * img_per_label + img_per_label)
        neg_idxs = np.setdiff1d(idx_arr, pos_idxs)

        compare_point = -1  # used to avoid redundancy in calculating SSE between anchor and all negative
        # Get all combination of Anchor and positive
        for indNum, (anc_idx, pos_idx) in enumerate(combinations(pos_idxs, 2)):
            if anc_idx != compare_point:
                compare_point += 1
                # Get the sum of squared distance between anchor and positive
                #
 
                anc_VS_neg_ndarr = tf.reduce_sum(
                                        tf.square(
                                            tf.subtract(
                                                tf.gather(batch_embedding, anc_idx),
                                                tf.gather(batch_embedding, neg_idxs)
                                             )
                                        ), 1
                                    )
        
            # Get the sum of squared distance between anchor and positive
            anc_VS_pos = tf.reduce_sum(
                            tf.square(
                                tf.subtract(
                                    tf.gather(batch_embedding, anc_idx),
                                    tf.gather(batch_embedding, pos_idx)
                                )
                            )
                        )
            
            # print ('anc_VS_pos anc_VS_pos', anc_VS_pos.eval())
            
            condition = tf.subtract(anc_VS_neg_ndarr, anc_VS_pos)
            hard_neg_idx = tf.where(condition < alpha)
            # print('hard_neg_idx hard_neg_idx ', hard_neg_idx.eval())

            # Randomly sample 1 record from the set of hard negative idx, and create a triplet
            if hard_neg_idx.eval().shape[0] > 0 :
                # print('hard_neg_idx hard_neg_idx ', hard_neg_idx)
                hard_neg_idx = tf.random_shuffle(hard_neg_idx)
                rnd_idx = tf.gather(hard_neg_idx, [0])
                print ('neg_idx neg_idx ', rnd_idx.eval())
                # Get triplet indexes in order to work on offline mode
                print (tf.cast(tf.gather(neg_idxs, rnd_idx), dtype=tf.int32).eval())
                batch_tripet_idx = tf.stack([anc_idx, pos_idx,
                                             tf.squeeze(tf.gather(neg_idxs, rnd_idx))])
                # print ('545454545454545 ', batch_tripet_idx.eval())
                # print ('')
                # print (break_point - cnt)
            if batch_tripet_idxs.get_shape().as_list()[0] == 0:
                batch_tripet_idxs = batch_tripet_idx
                # print('1111111111111111 ', batch_tripet_idxs.eval())
            else:
                # print ('12121212121212121 ', batch_tripet_idxs.get_shape().as_list())
                # print('12121212121212121 ', batch_tripet_idx.get_shape().as_list())
                batch_tripet_idxs = tf.stack([batch_tripet_idxs, batch_tripet_idx])
                # print('22222222222222222 ', batch_tripet_idxs.eval())
                
            # print ('vfvrgbrtbrbrtbrgbrt ', batch_tripet_idxs.eval())
    # print('TRIPLET INDEX ', batch_tripet_idx)

    return batch_tripet_idxs



        

#
#[[[0, 1, 39], [0, 2, 15], [0, 3, 22], [0, 4, 28], [0, 5, 14], [0, 6, 20], [0, 7, 38], [0, 8, 22], [0, 9, 15], [1, 2, 23], [1, 3, 31], [1, 4, 34], [1, 5, 39], [1, 6, 29], [1, 7, 35], [1, 8, 32], [1, 9, 21], [2, 3, 28], [2, 4, 22], [2, 5, 32], [2, 6, 13], [2, 7, 12], [2, 8, 16], [2, 9, 27], [3, 4, 29], [3, 5, 31], [3, 6, 26], [3, 7, 16], [3, 8, 28], [3, 9, 23], [4, 5, 35], [4, 6, 22], [4, 7, 32], [4, 8, 30], [4, 9, 23], [5, 6, 13], [5, 7, 30], [5, 8, 31], [5, 9, 29], [6, 7, 11], [6, 8, 23], [6, 9, 17], [7, 8, 12], [7, 9, 11], [8, 9, 36], [10, 11, 5], [10, 12, 8], [10, 13, 39], [10, 14, 8], [10, 15, 9], [10, 16, 24], [10, 17, 9], [10, 18, 20], [10, 19, 1], [11, 12, 33], [11, 13, 2], [11, 14, 39], [11, 15, 5], [11, 16, 38], [11, 17, 39], [11, 18, 2], [11, 19, 26], [12, 13, 4], [12, 14, 30], [12, 15, 26], [12, 16, 0], [12, 17, 4], [12, 18, 9], [12, 19, 25], [13, 14, 1], [13, 15, 25], [13, 16, 22], [13, 17, 31], [13, 18, 32], [13, 19, 20], [14, 15, 30], [14, 16, 21], [14, 17, 1], [14, 18, 27], [14, 19, 28], [15, 16, 35], [15, 17, 39], [15, 18, 2], [15, 19, 30], [16, 17, 8], [16, 18, 25], [16, 19, 28], [17, 18, 5], [17, 19, 3], [18, 19, 21], [20, 21, 13], [20, 22, 18], [20, 23, 37], [20, 24, 5], [20, 25, 19], [20, 26, 14], [20, 27, 30], [20, 28, 14], [20, 29, 6], [21, 22, 2], [21, 23, 2], [21, 24, 12], [21, 25, 10], [21, 26, 12], [21, 27, 39], [21, 28, 3], [21, 29, 14], [22, 23, 4], [22, 24, 39], [22, 25, 16], [22, 26, 11], [22, 27, 2], [22, 28, 19], [22, 29, 4], [23, 24, 10], [23, 25, 17], [23, 26, 30], [23, 27, 19], [23, 28, 30], [23, 29, 6], [24, 25, 36], [24, 26, 37], [24, 27, 32], [24, 28, 8], [24, 29, 15], [25, 26, 30], [25, 27, 17], [25, 28, 10], [25, 29, 35], [26, 27, 1], [26, 28, 6], [26, 29, 14], [27, 28, 3], [27, 29, 6], [28, 29, 17], [30, 31, 14], [30, 32, 25], [30, 33, 16], [30, 34, 18], [30, 35, 27], [30, 36, 28], [30, 37, 26], [30, 38, 4], [30, 39, 29], [31, 32, 22], [31, 33, 18], [31, 34, 5], [31, 35, 8], [31, 36, 2], [31, 37, 1], [31, 38, 29], [31, 39, 8], [32, 33, 16], [32, 34, 1], [32, 35, 1], [32, 36, 23], [32, 37, 13], [32, 38, 10], [32, 39, 29], [33, 34, 1], [33, 35, 24], [33, 36, 0], [33, 37, 23], [33, 38, 8], [33, 39, 5], [34, 35, 7], [34, 36, 19], [34, 37, 19], [34, 38, 1], [34, 39, 6], [35, 36, 6], [35, 37, 7], [35, 38, 1], [35, 39, 5], [36, 37, 12], [36, 38, 16], [36, 39, 17], [37, 38, 10], [37, 39, 28], [38, 39, 22]]]


#
# from random import randint
#
# def random_with_N_digits(n, how_many):
#     random.seed(8271)
#     f
#     range_start = 10**(n-1)
#     range_end = (10**n)-1
#     return randint(range_start, range_end)
#
#
# print (random_with_N_digits(4))