import tensorflow as tf
import numpy as np
from itertools import combinations


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


def getTriplets(batch_embedding, img_per_label, num_labels, alpha=0.01):
    batch_tripet_idx = []
    idx_arr = np.arange(len(batch_embedding))
    for i in np.arange(num_labels):
        pos_idxs = np.arange( i *img_per_label , i * img_per_label +img_per_label)
        neg_idxs = np.setdiff1d(idx_arr, pos_idxs)
        # print(pos_idxs)
        # print('')
        # print(neg_idxs)
        compare_point = -1  # used to avoid redundancy in calculating SSE between anchor and all negative
        # Get all combination of Anchor and positive
        # print ('######################################')
        for anc_idx, pos_idx in combinations(pos_idxs, 2):
            if anc_idx != compare_point:
                compare_point += 1
                # Get the sum of squared distance between anchor and positive
                anc_VS_neg_ndarr = np.sum(np.square(
                        batch_embedding[anc_idx] - batch_embedding[neg_idxs]), 1
                )
                # print (anc_VS_neg_ndarr)
            
            # Get the sum of squared distance between anchor and positive
            anc_VS_pos = np.sum(
                    np.square(batch_embedding[anc_idx] - batch_embedding[pos_idx]))
            # print ('anc_VS_pos anc_VS_pos', anc_VS_pos)
            
            hard_neg_idx = np.where(anc_VS_neg_ndarr - anc_VS_pos < alpha)[0]
            
            
            # Randomly sample 1 record from the hard negative idx, and create a triplet
            if len(hard_neg_idx ) >0:
                # print('hard_neg_idx hard_neg_idx ', hard_neg_idx)
                np.random.shuffle(hard_neg_idx)
                rnd_idx = hard_neg_idx[0]
                # print ('neg_idx neg_idx ', neg_idxs[rnd_idx])
                
                # Get triplet indexes in order to work on offline mode
                batch_tripet_idx.append([anc_idx, pos_idx, neg_idxs[rnd_idx]])
                # print (anc_idx,pos_idx)
                # print ('')
                # print (break_point - cnt)
    # print('TRIPLET INDEX ', batch_tripet_idx)
    
    return batch_tripet_idx


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