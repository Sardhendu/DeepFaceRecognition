import tensorflow as tf
import numpy as np
from itertools import combinations
import logging


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
                np.random.shuffle(hard_neg_idx)
                rnd_idx = hard_neg_idx[0]
                
                # Get triplet indexes in order to work on offline mode
                batch_tripet_idx.append([anc_idx, pos_idx, neg_idxs[rnd_idx]])
    logging.info('TRIPLETS %s == %s', str(len(batch_tripet_idx)), str(batch_tripet_idx))
    return [batch_tripet_idx]



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




# np.random.seed(327)
# batch_embedding = np.random.rand(40, 2)
# batch_size, num_embeddings = batch_embedding.shape
# img_per_label = 10
# num_labels = int( batch_size /img_per_label)
#
# alpha = 0.01
#
#
# batch_tripet_idx = getTriplets(batch_embedding, img_per_label, num_labels)
#
# print (batch_tripet_idx)