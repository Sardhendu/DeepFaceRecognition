from __future__ import division, print_function, absolute_import

import logging
import os

import numpy as np
import tensorflow as tf
from data_transformer.data_io import getPickleFile

import config
import cv2
from config import path_dict
from collections import defaultdict
from data_transformer.preprocess import Preprocessing
from nn.load_params import getWeights
from train_test.classifier import SVM
from data_transformer.detect_extract_faces import detect_extract_faces
from train_test.model import getEmbeddings, trainEmbeddings, summaryBuilder


def get_pretrained_weights():
    moduleWeightDict = getWeights(path_dict['inception_nn4small_weights_path'])
    return moduleWeightDict


class PlotLabeledImages():
    def draw_rectangle(self, img, x, y, w, h):
        print('Input image shape: ', img.shape)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        return img
    
    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img
    
    def get_save_labeled_image(self, img_path, rect_faces, text_labels, label_prob):
        img = cv2.imread(img_path)
        for rect, text, prob in zip(rect_faces, text_labels, label_prob):
            (x, y, w, h) = rect
            img = self.draw_rectangle(img, x, y, w, h)
            img = self.draw_text(img, text+' '+str(round(prob*100, 2))+'%', x, y)
        cv2.imwrite(os.path.join(path_dict['face_detection_labeled_test_path'],
                                 os.path.basename(img_path)), img)
        return img



class DeepFaceRecognition():
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
        self.weights = None
        params_keys = params.keys()
        #         print (hyper_params)
        if 'learning_rate_override' in params_keys:
            self.myNet['learning_rate'] = params['learning_rate_override']
        
        if 'init_finetune_weight' in params_keys:
            self.init_finetune_weight = params['init_finetune_weight']
        
        if 'write_tensorboard_summary' in params_keys:
            self.write_tensorboard_summary = params['write_tensorboard_summary']
        
        if 'save_for_analysis' in params_keys:
            self.save_for_analysis = params['save_for_analysis']
        
        if 'use_checkpoint' in params_keys:
            self.use_checkpoint = params['use_checkpoint']
        
        if 'save_checkpoint' in params_keys:
            self.save_checkpoint = params['save_checkpoint']

        if 'which_fold' in params_keys:
            if params['which_fold'] >= 10:
                raise ValueError('The which_fold values runs from 0 through 9')
            self.nFold = params['which_fold']
        
        if 'numEpochs' in params_keys:
            self.numEpochs = params['numEpochs']
        
        if 'which_eopch_to_save' in params_keys:
            self.which_eopch_to_save = params['which_eopch_to_save']
        
        if 'checkpoint_file_name' in params_keys:
            self.checkpoint_file_name = params['checkpoint_file_name']
        
        if 'batch_file_name' in params_keys:
            self.batch_file_name = params['batch_file_name']
        
        if 'test_image_path_arr' in params_keys:
            self.test_image_path_arr = test_image_path_arr
        
        _, _, self.labelDict = getPickleFile(
                folderPath=path_dict['parent_path'], picklefileName='training_imgarr.pickle', getStats=True
        )

        
    def accuracy(self, y, y_hat):
        return np.mean(np.equal(y_hat, y))


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
    
    def extract_faces(self, test_image_path_arr):
        '''
             test_faces: If the input image has two people then the test_faces would be a list of two nd array where
             each ndarray represent the face of the person

             rect_faces: If the input image has two people then the rect_faces would be a list of two array where
             each array represent the coordinate or rather the x,y coordinate and h and w of the extracted faces.
        '''
        test_image_dict = defaultdict(lambda: defaultdict())
        for num, image_path in enumerate(test_image_path_arr):
            image_name = os.path.basename(image_path).split('.')[0]
            print(image_path)
            print(path_dict['face_extracted_test_path'])
            print(path_dict['face_detection_test_path'])
            test_faces, rect_faces = detect_extract_faces(image_path,
                                                          os.path.join(path_dict['face_extracted_test_path'],
                                                                       '%s.jpg' % str(image_name)),
                                                          os.path.join(path_dict['face_detection_test_path'],
                                                                       '%s.jpg' % str(image_name)),
                                                          store=True)
            test_image_dict[image_name]['test_faces'] = test_faces
            test_image_dict[image_name]['rect_faces'] = rect_faces
            test_image_dict[image_name]['image_path'] = image_path
        return test_image_dict
    
    # Both the train and test class uses this method
    def runPreprocessor(self, dataIN, sess):
        logging.info('INITIATING PREPROCESSING.................')
        preprocessedData = np.ndarray(shape=(dataIN.shape), dtype='float32')
        for numImage in np.arange(dataIN.shape[0]):
            feed_dict = {
                self.preprocessGraphDict['imageIN']: dataIN[numImage, :]
            }
            preprocessedData[numImage, :] = sess.run(self.preprocessGraphDict['imageOUT'], feed_dict=feed_dict)
        return preprocessedData
    
    def create_graph(self, weights):
        # RESET AND CREATE THE GRAPH
        self.preprocessGraphDict = Preprocessing().preprocessImageGraph(imageShape=self.myNet["image_shape"])
        self.trn_embed_graph = trainEmbeddings(weights, init_wght_type=self.init_finetune_weight)


class Train(DeepFaceRecognition):
    def _init__(self, myNet, embeddingType='finetune', params={}):
        DeepFaceRecognition.__init__(self, myNet, embeddingType=embeddingType, params=params)
    
    def train(self, trnX_, trnY_, sess):
        '''
            1. Make the use of getEmbedding to get the graph with last layer parameter updated with the
            fine tuned weights.
            2. Get the new embedding for batch/epoch using the computation graph
            3. Use the embeddings as feature for a classifier (svm/softmax)
            4. Classify faces using the new embeddings.
        '''
        logging.info('Fetch Embeddings for Train: Performing SVM classification on the preprocessed train data')
        # CREATE THE EMBEDDING GRAPH WITH LATEST WEIGHTS
        embeddingGraph = getEmbeddings(self.myNet['image_shape'], self.weights)
        trnX_pp = self.runPreprocessor(dataIN=trnX_, sess=sess)
        feed_dict = {embeddingGraph['inpTensor']: trnX_pp}
        embeddings = sess.run(embeddingGraph['embeddings'], feed_dict=feed_dict)
        
        logging.info('Training Embeddings shape %s', embeddings.shape)
        
        obj_svm = SVM()
        obj_svm.train(embeddings, labels=trnY_, model_name='final_model_Fold_%s' % (str(self.nFold)))
        train_labels, train_label_prob = obj_svm.classify(embeddings, model_name='final_model_Fold_%s' % (str(
                self.nFold)))
        return train_labels, train_label_prob
    
    def cvalid(self, cvX_, sess):
        logging.info('Fetch Embeddings for CV: Performing SVM classification on the preprocessed Validation data')

        # CREATE THE EMBEDDING GRAPH WITH LATEST WEIGHTS
        embeddingGraph = getEmbeddings(self.myNet['image_shape'], self.weights)
        cvX_pp = self.runPreprocessor(dataIN=cvX_, sess=sess)
        feed_dict = {embeddingGraph['inpTensor']: cvX_pp}
        embeddings = sess.run(embeddingGraph['embeddings'], feed_dict=feed_dict)
        
        logging.info('Cross validation Embeddings shape %s', embeddings.shape)
        obj_svm = SVM()
        cv_labels, cv_label_prob = obj_svm.classify(embeddings, model_name='final_model_Fold_%s' % (str(self.nFold)))
        return cv_labels, cv_label_prob
    
    def accuracy(self, y, y_hat):
        return np.mean(np.equal(y_hat, y))
    
    def exec_batch(self, sess):
        feed_dict ={}
        batch_loss = 0
        for batchNum, batchX in enumerate(self.trnX[0:len(self.trnX), :]):
            logging.info('RUNNING BATCH %s for shape = %s', str(batchNum + 1), str(batchX.shape))
        
            # Step1 : Preprocess the Data
            preprocessedData = self.runPreprocessor(dataIN=batchX, sess=sess)
            feed_dict = {self.trn_embed_graph['inpTensor']: preprocessedData}
            opt, batch_loss, lr = sess.run([self.trn_embed_graph['optimizer'],
                                            self.trn_embed_graph['triplet_loss'],
                                            self.trn_embed_graph['learning_rate']],
                                           feed_dict=feed_dict)
            logging.info('Learning Rate (Current) is: %s', str(lr))
    
        # Store the summary, and print the loss, accuracy after every epoch or for every batch
        if self.write_tensorboard_summary:
            smry = sess.run(self.mergedSummary, feed_dict=feed_dict)
            self.writer.add_summary(smry, self.epoch)
        
        return batch_loss
            
    def exec_epoch(self, sess):
        tr_acc = 0
        cv_acc = 0
        for epoch in np.arange(self.numEpochs):
            self.epoch = epoch + 1
            logging.info('RUNNING : %s EPOCH ........................', str(self.epoch))
            # Below loop will minimize the triplet loss and update the parameters

            batch_loss = self.exec_batch(sess)
        
            print("Fold= " + str(self.nFold) +
                  ", Epoch= " + str(self.epoch) +
                  ", Loss= " + "{:.6f}".format(batch_loss))
        
            # Now that we have updated our parameters (weights and biases), we would
            # fetch the embeddings using the updated parameter and train-test model
            # to get an accuracy. Accuracy per epoch is now a good way to go
            self.setNewWeights(sess)  # replace the last layer's inception weights with leared finetuned weights
        
            # TRAIN, GET TRAINING PREDICTION AND ACCURACY
            trnX_ = self.trnX.reshape(-1, self.trnX.shape[2], self.trnX.shape[3], self.trnX.shape[4])  # accumulate all
            # batches
            trnY_ = self.trnY.flatten()
            train_labels, _ = self.train(trnX_, trnY_, sess)
            tr_acc = self.accuracy(y=trnY_, y_hat=train_labels)
            print("Fold: %s, Train acc = %s " % (str(self.nFold), str(tr_acc)))
        
            # GET CROSS VALIDATION PREDICTION AND ACCURACY
            cv_labels, cv_pred_prob = self.cvalid(self.cvX, sess)
            logging.info('Predicted Labels : %s', str(cv_labels))
            logging.info('Predicted Probabilities : %s', str(cv_pred_prob))
        
            cv_acc = self.accuracy(y=self.cvY, y_hat=cv_labels)
            print("Fold: %s, CV acc = %s " % (str(self.nFold), str(cv_acc)))
        
            # if self.epoch in self.which_eopch_to_save and self.save_for_analysis:
            #     logging.info('Saving the output probaboilities for analysis ....')
            #     save_prediction_analysis(cv_act=cvY, cv_hat=cv_labels, cv_hat_prob=cv_pred_prob,
            #                              fold=self.nFold, epoch=self.epoch, cvBatch_num = self.cv_batch_idx)
        return tr_acc, cv_acc

    def sess_exec(self):
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Retrieve the checkpoint from previous run:
            checkpoints = [ck for ck in os.listdir(path_dict['checkpoint_path']) if ck != '.DS_Store']
            if self.use_checkpoint and len(checkpoints) > 0 :
                checkpoint_path = os.path.join(path_dict['checkpoint_path'],
                                               self.checkpoint_file_name + '.ckpt'
                                               if len(self.checkpoint_file_name.split('.')) == 1
                                               else self.checkpoint_file_name)
                saver.restore(sess, checkpoint_path)
            
            # GET THE SUMMARY OUTPUT OF TENSOR BOARD
            if self.write_tensorboard_summary:
                self.mergedSummary, self.writer = summaryBuilder(sess, path_dict["summary_path"])

            tr_acc, cv_acc = self.exec_epoch(sess)
            
            if self.save_checkpoint:
                logging.info('SAVING CHECKPOINT TO BE USED BY THE TEST DATA ....')
                checkpoint_path = os.path.join(path_dict['checkpoint_path'],
                                               self.checkpoint_file_name + '.ckpt'
                                               if len(self.checkpoint_file_name.split('.')) == 1
                                               else self.checkpoint_file_name)
                saver.save(sess, checkpoint_path)
            
            if self.write_tensorboard_summary:
                self.writer = tf.summary.FileWriter(path_dict["summary_path"], sess.graph)
                self.writer.close()
        return tr_acc, cv_acc
    
    def fit_save(self, weightsIN):
        self.weights = weightsIN
        
        # CREATE TEH TRAIN PREPROCESSING AND TRAIN GRAPH
        self.create_graph(self.weights)
        
        # GET THE BATCH DATA FROM THE DISK
        dataX, dataY, labelDict = getPickleFile(
                folderPath=path_dict['batchFolderPath'], picklefileName=self.batch_file_name, getStats=True
        )
        trnBatch_idx = [list(np.setdiff1d(np.arange(len(dataX)), np.array(i))) for i in np.arange(len(dataX))]
        cvBatch_idx = [i for i in np.arange(len(dataX))]
        trn_batch_idx = trnBatch_idx[self.nFold]
        cv_batch_idx = cvBatch_idx[self.nFold]
        print('Train batch indices :', trn_batch_idx)
        print('Cross validation batch indice : ', cv_batch_idx)
        logging.info('dataX.shape = %s, dataY.shape = %s', str(dataX.shape), str(dataY.shape))
        
        # NOTE WE HAVE TO RESET THE WEIGHTS to the Inception weights every FOLD

        # We reset all the seed indexes to ensure that all the weights/triplet selection for every fold
        # are iniitalized with the save start value for each fold
        config.weight_seed_idx = 0
        config.triplet_seed_idx = 0
        config.preprocess_seed_idx = 0
        #############   MAIN CALL START
        self.cv_batch_idx = cv_batch_idx
        print('Training Batch Numbers ', trn_batch_idx)
        print('CV Batch Number ', self.cv_batch_idx)
        
        logging.info('RUNNING : %s FOLD ...........................', str(self.nFold))
        self.trnX = dataX[trn_batch_idx, :]
        self.trnY = dataY[trn_batch_idx, :]
        self.cvX = dataX[cv_batch_idx, :]
        self.cvY = dataY[cv_batch_idx, :]
        logging.info('trnX.shape = %s, trnY.shape = %s, cvX.shape = %s, cvY.shape = %s',
                     str(self.trnX.shape), str(self.trnY.shape), str(self.cvX.shape), str(self.cvY.shape))
        
        
        # EXECUTE THE SESSION FOR THE CURRENT FOLD
        tr_acc, cv_acc = self.sess_exec()
        
        return tr_acc, cv_acc


class Test(DeepFaceRecognition):
    
    def _init__(self, myNet, embeddingType='finetune', params={}):
        DeepFaceRecognition.__init__(self, myNet, embeddingType=embeddingType, params=params)
        
    def test_exec_sess(self, input_image_arr):
        # RESETTING THE DEFAULT SESSION
        tf.reset_default_graph()
        
        # CREATE THE PREPROCESSING GRAPH AND TRAIN GRAPH, (TRAIN GRAPH IS NOT NEEDED) BUT WE OBTAIN IT TO GET THE
        # WEIGHT VARIABLES, TO BE ABLE TO RUN tf.train.Saver(
        self.create_graph(self.weights)
        
        # CREATE THE EMBEDDING GRAPH
        embeddingGraph = getEmbeddings(self.myNet['image_shape'], self.weights)
        
        # RESET AND CREATE THE GRAPH
        config.weight_seed_idx = 0
        logging.info('Test input_image_arr.shape = %s', str(input_image_arr.shape))
    
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            checkpoint_path = os.path.join(path_dict['checkpoint_path'],
                                           self.checkpoint_file_name + '.ckpt'
                                           if len(self.checkpoint_file_name.split('.')) == 1
                                           else self.checkpoint_file_name)
            # RESTORE THE CHECK POINT
            saver.restore(sess, checkpoint_path)
        
            # SET PRE-TRAINED WEIGHTS FROM THE INITIALIZED INCEPTION WEIGHTS AND SET
            # FINE TUNED WEIGHTS FROM THE CHECKPOINT
            self.setNewWeights(sess)
        
            
            test_pp = self.runPreprocessor(dataIN=input_image_arr, sess=sess)
            feed_dict = {embeddingGraph['inpTensor']: test_pp}
            
            embeddings = sess.run(embeddingGraph['embeddings'], feed_dict=feed_dict)
            
            logging.info('Test Image Embeddings shape %s', embeddings.shape)
            obj_svm = SVM()
            cv_labels, cv_label_prob = obj_svm.classify(embeddings, model_name='final_model_Fold_%s' % (str(self.nFold)))
            
            print(cv_labels)
            print(" Probabilities acc = %s " % str(cv_label_prob))
    
        label_names = [self.labelDict[str(i)].upper() for i in cv_labels]
        return label_names, cv_label_prob
    
    
    def predict(self, test_image_path_arr, weightsIN):
        # FIRST INITIALIZE THE WEIGHTS BY PRETRAINED WEIGHTS
        self.weights = weightsIN
        
        test_image_dict = self.extract_faces(test_image_path_arr)
        labeled_image_arr = []
        for image, img_features in test_image_dict.items():
            face_arr = img_features['test_faces']
            rect_face_arr = img_features['rect_faces']
            print(face_arr.shape, rect_face_arr.shape)
        
            label_names, cv_label_prob = self.test_exec_sess(face_arr)
        
            labeled_image_arr.append(
                PlotLabeledImages().get_save_labeled_image(img_features['image_path'], rect_faces=rect_face_arr,text_labels=label_names, label_prob=cv_label_prob))
    
        return labeled_image_arr
    

debugg = True

if debugg:
    from config import myNet
    
    test_image_path_arr = [
        '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/output_data_faces/face_snapshot/img2.jpg',
        '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/output_data_faces/face_snapshot/img1.jpg',
        '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/output_data_faces/face_snapshot/img3.jpg'
    ]
    # myNet['triplet_selection_alpha'] = 0.1
    myNet['triplet_selection_alpha'] = 0.09
    myNet['triplet_loss_penalty'] = 0.1
    params = dict(learning_rate_override=0.0001,
                  init_finetune_weight='random',
                  use_checkpoint=False,
                  save_checkpoint=True,
                  write_tensorboard_summary=False,
                  save_for_analysis=False,
                  which_fold=5,  # In actuality
                  numEpochs=10,
                  which_eopch_to_save=[5, 8, 9, 10],
                  batch_file_name='ext_rsz_distinct_stratified_batches.pickle',
                  checkpoint_file_name='ext_rsz_distinct_stratified_model'
                  )
    moduleWeightDict = getWeights(path_dict['inception_nn4small_weights_path'])
    train_obj = Train(myNet=myNet, embeddingType='finetune', params=params)
    test_obj = Test(myNet=myNet, embeddingType='finetune', params=params)
    tr_acc, cv_acc = train_obj.fit_save(moduleWeightDict)
    labeled_image_arr = test_obj.predict(test_image_path_arr, moduleWeightDict)
    
    