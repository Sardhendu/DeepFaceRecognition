


import logging
import os

from config import myNet
from config import path_dict
from dsdsdsdsd.data_prep import DataIO, genDistinctStratifiedBatches
from data_transformer.data_prep import imageToArray
from train_test.train_save_model import Train  # , Test

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def create_training_data_wrapper():
    objDP = DataFormatter(path_dict['parent_path'], 'training')
    objDP.createResizedData()
    dataX, dataY, labelDict, person_name_image_num_info = objDP.imageToArray(get_stats=True)
    
    DataFormatter.dumpPickleFile(dataX, dataY, labelDict,
                                 folderPath=os.path.join(path_dict['data_model_path']),
                                 picklefileName='training_imgarr.pickle')
    
    DataFormatter.dumpCSVFile(person_name_image_num_info,
                              folderPath=os.path.join(path_dict['data_model_path']),
                              csvfileName='person_name_image_num_info.csv')


def create_verification_data_wrapper():
    objDP = DataFormatter(path_dict['parent_path'], 'verification')
    objDP.createResizedData()
    dataX, dataY, labelDict = objDP.imageToArray()
    DataFormatter.dumpPickleFile(dataX, dataY, labelDict,
                                 folderPath=os.path.join(path_dict['data_model_path']),
                                 picklefileName='verification_imgarr.pickle')


def create_batches_wrapper():
    trainX, trainY, trainLabelDict = DataIO.getPickleFile(path_dict['data_model_path'],
                                                          'training_imgarr.pickle')
    verX, verY, verLabelDict = DataIO.getPickleFile(path_dict['data_model_path'],
                                                    'verification_imgarr.pickle')
    print(trainX.shape, trainY.shape)
    print(verX.shape, verY.shape)
    genDistinctStratifiedBatches(trainX, trainY,
                                 fileName='distinct_stratified_batches.pickle')


def train():
    '''
        It seems from analysing the file (RecognitionTuneParams) that the hyperparameter triplet_selection = 0.05 does better in overall cross validation average accuracy and learning rate 0.0001 with exponential details. Other
        parameters are tested and stores in the config file. We don't need to touch them. Lets use the best model parameters and train, store our model trainable (finetune eligible) parameters.
    '''
    
    myNet['triplet_selection_alpha'] = 0.1
    params = dict(learning_rate_override=0.0001,
                  use_checkpoint=False,
                  write_tensorboard_summary = True,
                  save_for_analysis=True,
                  which_fold = 1,
                  numEpochs=10,
                  which_eopch_to_save=[5, 8, 9, 10],
                  batch_file_name = 'distinct_stratified_batches.pickle',
                  checkpoint_file_name='distinct_stratified_model')
    
    train_test_obj = Train(myNet=myNet, embeddingType='finetune', params=params)
    tr_acc, cv_acc = train_test_obj.train_save_model()

    print('trained_fold = %s, trained_epochs = %s, train_accuracy = %s, cross_validation_accuracy = %s' % (
        str(params['which_fold']), str(params['numEpochs']), str(tr_acc), str(cv_acc)))
    
def test():
    pass



train()