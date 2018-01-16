from __future__ import division, print_function, absolute_import

import os
import numpy as np
import tensorflow as tf
import logging
from data_transformer.data_formatter import DataFormatter
from data_transformer.preprocess import Preprocessing

from nn.load_params import layer_name, convShape, getWeights
from train_test.model import trainModel_FT, getEmbeddings, trainEmbeddings, summaryBuilder
from train_test.classify import SVM

import config
from config import path_dict



def get_pretrained_weights():
    moduleWeightDict = getWeights(path_dict['inception_nn4small_weights_path'])
    return moduleWeightDict

    

class TrainTest():
    '''
    # This module would train the network for the given parameters and store the weights (create checkpoints) in the
    # disk.
    # This module would also provide a training accuracy and cross validation accuracy, given the fold you select. For
    # example if the input which_fold = 1, then the model will only be trained on first 9 batch and would be
    # validated on
    #  the 10th batch.
    # Lately, for testing purpose, the module will pick up the checkpoint and provide the prediction
    '''
    def __init__(self, myNet, embeddingType='finetune', params={}):
        self.myNet = myNet
        self.embeddingType = embeddingType
        
        #         print (hyper_params)
        if 'learning_rate_override' in params.keys():
            self.myNet['learning_rate'] = params['learning_rate_override']

        self.write_tensorboard_summary = params['write_tensorboard_summary']
        self.save_for_analysis = params['save_for_analysis']
        self.use_checkpoint = params['use_checkpoint']
        self.nFold = params['which_fold'] - 1
        self.numEpochs = params['numEpochs']
        self.which_eopch_to_save = params['which_eopch_to_save']
        self.checkpoint_file_name = params['checkpoint_file_name']
        self.batch_file_name = params['batch_file_name']
    
    def runPreprocessor(self, dataIN, sess):
        preprocessedData = np.ndarray(shape=(dataIN.shape), dtype='float32')
        for numImage in np.arange(dataIN.shape[0]):
            feed_dict = {
                self.preprocessGraphDict['imageIN']: dataIN[numImage, :]
            }
            preprocessedData[numImage, :] = sess.run(self.preprocessGraphDict['imageOUT'],
                                                     feed_dict=feed_dict)
        return preprocessedData
    
    def resetWeights(self, weightsIN):
        logging.info('RESETTING WEITHGS WITH PRE-TRAINED WEIGHTS .........')
        self.weights = weightsIN
    
    def setNewWeights(self, sess):
        logging.info('UPDATING WEITHGS WITH FINETUNED WEIGHTS .........')
        #         trainableVars = tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        if self.embeddingType == 'finetune':
            for learned_vars in config.finetune_variables:
                scope, name = learned_vars.split(':')[0].split('/')
                if len(self.weights[scope][name]) != 0:
                    var_ = sess.run(learned_vars)
                    logging.info('Updating param with scope %s and name %s and shape %s with shape %s',
                                 str(scope), str(name), str(self.weights[scope][name].shape), str(var_.shape))
                    self.weights[scope][name] = var_
                else:
                    raise ValueError('It seems that the scope %s or variable %s didnt exist in the dictionary ' % (
                        str(scope), str(name)))
    
    def train(self, trnX_, trnY_, sess):
        '''
            1. Make the use of getEmbedding to get the graph with last layer parameter updated with the
            fine tuned weights.
            2. Get the new embedding for batch/epoch using the computation graph
            3. Use the embeddings as feature for a classifier (svm/softmax)
            4. Classify faces using the new embeddings.
        '''
        trainEmbedGraph = getEmbeddings(self.myNet['image_shape'], self.weights)
        embeddings = sess.run(trainEmbedGraph['embeddings'],
                              feed_dict={trainEmbedGraph['inpTensor']: trnX_})
        logging.info('Training Embeddings shape %s', embeddings.shape)
        obj_svm = SVM()
        obj_svm.train(embeddings, labels=trnY_,
                      model_name='nFold_%s_batch_%s' % (str(self.nFold), str(self.epoch)))
        train_labels, train_label_prob = obj_svm.classify(embeddings,
                                                          model_name='nFold_%s_batch_%s' % (
                                                              str(self.nFold), str(self.epoch)))
        return train_labels, train_label_prob
    
    def cvalid(self, cvX_, sess):
        embedGraph = getEmbeddings(self.myNet['image_shape'], self.weights)
        embeddings = sess.run(embedGraph['embeddings'],
                              feed_dict={embedGraph['inpTensor']: cvX_})
        logging.info('Cross validation Embeddings shape %s', embeddings.shape)
        obj_svm = SVM()
        cv_labels, cv_label_prob = obj_svm.classify(embeddings,
                                                    model_name='nFold_%s_batch_%s' % (str(self.nFold), str(self.epoch)))
        return cv_labels, cv_label_prob
    
    def accuracy(self, y, y_hat):
        return np.mean(np.equal(y_hat, y))
    
    #     def test(self, tstGraph, testBatch, sess):
    #         # METHOD 2: TO get weights is form of Tensors
    #         a = saver.restore(sess, os.path.join(checkpoint_path, "model.ckpt"))
    #         trainableVars = tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    #         testDict = getFineTunedEmbeddings([96,96,3], moduleWeightDict, trainableVars, sess)
    #         embeddings = sess.run([tstGraph['output']], feed_dict={'inpTensor':testBatch})
    #         return embeddings
    
    
    def sess_exec(self, trnX, trnY, cvX, cvY):
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Save the checkpoint of the run
            checkpoints = [ck for ck in os.listdir(path_dict['checkpoint_path']) if ck != '.DS_Store']
            if len(checkpoints) > 0 and self.use_checkpoint:
                saver.restore(sess, os.path.join(path_dict['checkpoint_path'], self.checkpoint_file_name))
            
            # Get the summary output for tensorboard
            if self.write_tensorboard_summary:
                self.mergedSummary, self.writer = summaryBuilder(sess, path_dict["summary_path"])
            
            tr_acc_arr = []
            cv_acc_arr = []
            tr_acc = 0
            cv_acc = 0
            feed_dict = {}
            for epoch in np.arange(self.numEpochs):
                self.epoch = epoch + 1
                logging.info('RUNNING : %s EPOCH ........................', str(self.epoch))
                # Below loop will minimize the triplet loss and update the parameters
                for batchNum, batchX in enumerate(trnX[0:len(trnX), :]):
                    logging.info('RUNNING BATCH %s for shape = %s', str(batchNum + 1), str(batchX.shape))
                    
                    # Step1 : Preprocess the Data
                    preprocessedData = self.runPreprocessor(dataIN=batchX, sess=sess)
                    
                    # Since we improve on our previous prediction, there can be cases where the network has learned a
                    #  good enough
                    # decision boundary (for a batch) and is unable to find hard negative for the triplet selection.
                    # In such a case
                    # the network would return an empty array, which would raise a run time exception during the
                    # graph is computed.
                    # For such cases we would except an exception, and let the graph proceed.
                    
                    try:
                        feed_dict = {self.trn_embed_graph['inpTensor']: preprocessedData}
                        opt, batch_loss, lr = sess.run([self.trn_embed_graph['optimizer'],
                                                        self.trn_embed_graph['triplet_loss'],
                                                        self.trn_embed_graph['learning_rate']],
                                                       feed_dict=feed_dict)
                    except Exception:
                        logging.info(
                                'Exception Raised! Check the log file and confirm if the exception is becasue of empty '
                                'triplet array. If not then debugg it :)')
                        logging.info("Fold = %s, Epoch = %s, Loss = %s",
                                     str(self.nFold), str(self.epoch), "{:.6f}".format(batch_loss))
                
                # Store the summary, and print the loss, accuracy after every epoch or for every batch
                if self.write_tensorboard_summary:
                    smry = sess.run(self.mergedSummary, feed_dict=feed_dict)
                    self.writer.add_summary(smry, self.epoch)
                
                print("Fold= " + str(self.nFold) +
                      ", Epoch= " + str(self.epoch) +
                      ", Loss= " + "{:.6f}".format(batch_loss))
                
                # Now that we have updated our parameters (weights and biases), we would
                # fetch the embeddings using the updated parameter and train-test model
                # to get an accuracy. Accuracy per epoch is now a good way to go
                self.setNewWeights(sess)  # replace the last layer's inception weights with leared finetuned weights
                
                # TRAIN, GET TRAINING PREDICTION AND ACCURACY
                trnX_ = trnX.reshape(-1, trnX.shape[2], trnX.shape[3], trnX.shape[4])  # accumulate all batches
                trnY_ = trnY.flatten()
                train_labels, _ = self.train(trnX_, trnY_, sess)
                tr_acc = self.accuracy(y=trnY_, y_hat=train_labels)
                tr_acc_arr.append(tr_acc)
                print("Fold: %s, Train acc = %s " % (str(self.nFold), str(tr_acc)))
                
                # GET CROSS VALIDATION PREDICTION AND ACCURACY
                cv_labels, cv_pred_prob = self.cvalid(cvX, sess)
                cv_acc = self.accuracy(y=cvY, y_hat=cv_labels)
                cv_acc_arr.append(cv_acc)
                print("Fold: %s, CV acc = %s " % (str(self.nFold), str(cv_acc)))

            # Write the weights, to the file to use it later for testing purpose
            saver.save(sess, os.path.join(path_dict['checkpoint_path'], self.checkpoint_file_name))

            if self.write_tensorboard_summary:
                self.writer = tf.summary.FileWriter(path_dict["summary_path"], sess.graph)
                self.writer.close()
        return tr_acc, cv_acc, tr_acc_arr, cv_acc_arr
    
    
    def train_save_model(self):
        # Get the model weights
        self.weights = get_pretrained_weights()
        
        # GET THE BATCH DATA FROM THE DISK
        dataX, dataY, labelDict = DataFormatter.getPickleFile(
                folderPath=path_dict['batchFolderPath'], picklefileName=self.batch_file_name, getStats=True
        )
        trnBatch_idx = [list(np.setdiff1d(np.arange(len(dataX)), np.array(i))) for i in np.arange(len(dataX))]
        cvBatch_idx = [i for i in np.arange(len(dataX))]
        
        logging.info('dataX.shape = %s, dataY.shape = %s', str(dataX.shape), str(dataY.shape))

        
        # We reset all the seed indexes to ensure that all the weights/triplet selection for every fold
        # are iniitalized with the save start value for each fold
        config.weight_seed_idx = 0
        config.triplet_seed_idx = 0
        config.preprocess_seed_idx = 0

        # Get training and validation dataset, to avoid surprises with the result
        logging.info('RUNNING : %s FOLD ...........................', str(self.nFold))
        trnX = dataX[trnBatch_idx[self.nFold], :]
        trnY = dataY[trnBatch_idx[self.nFold], :]
        cvX = dataX[cvBatch_idx[self.nFold], :]
        cvY = dataY[cvBatch_idx[self.nFold], :]
        logging.info('trnX.shape = %s, trnY.shape = %s, cvX.shape = %s, cvY.shape = %s',
                     str(trnX.shape), str(trnY.shape), str(cvX.shape), str(cvY.shape))
        
        # CREATE THE GRAPH
        self.trn_embed_graph = trainEmbeddings(self.weights, init_wght_type='random')
        self.preprocessGraphDict = Preprocessing().preprocessImageGraph(imageShape=self.myNet["image_shape"])
        
        # EXECUTE THE SESSION FOR THE CURRENT FOLD
        tr_acc, cv_acc, tr_acc_arr, cv_acc_arr = self.sess_exec(trnX, trnY, cvX, cvY)
    
        return tr_acc, cv_acc,




##################################################




