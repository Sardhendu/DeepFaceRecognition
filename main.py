import logging
import os

import config
from config import path_dict

from nn.load_params import getWeights
from train_test.run import Test, Train

from data_transformer.generate_data import extract_resize_faces_for_training
from data_transformer.data_prep import imageToArray, genDistinctStratifiedBatches, genRandomStratifiedBatches
from data_transformer.data_io import dumpPickleFile, dumpCSVFile, getPickleFile

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")



data_prep = True
train_test_model = True

############################  PREPARE DATA  ############################
# EXTRACT FACES FORM LARGE IMAGES AND SUMPS TO THE DISK
# USES THE FACE EXTRACTED DATA TO CREATE 10-FOLD BATCHES AND DUMPS AS PICKLE
# FILE FOR THE TRAIN TEST MODULE TO MAKE USE OF
if data_prep:
    transform_type = 'extract_resize'  # 'only_resize'
    extract_resize_faces = False
    stack_face_into_ndarray = False
    create_distinct_stratified_batches = True
    create_random_stratified_batches = False
    
    if transform_type == 'extract_resize':
        input_faces_path = config.path_dict['face_extracted_path']
        pickle_file_name = 'ext_rsz_training_imgarr.pickle'
        batch_file_name = 'ext_rsz_distinct_stratified_batches.pickle'
    elif transform_type == 'only_resize':
        input_faces_path = config.path_dict['face_snapshot_resized_path']
        pickle_file_name = 'only_rsz_training_imgarr.pickle'
        batch_file_name = 'only_rsz_distinct_stratified_batches.pickle'
    else:
        raise ValueError('No proper transform_type provided')
    
    if create_distinct_stratified_batches and create_random_stratified_batches:
        raise ValueError('Only one type of batch creation permitted for one run')
    
    if extract_resize_faces:
        # GENERATE FACES (EXTRACT AND RESIZE OR JUST RESIZE)
        extract_resize_faces_for_training(transform_type=transform_type)
    
    if stack_face_into_ndarray:
        # CONVERT THE FACES INTO NDARRAYS
        dataX, dataY, labelDict, person_name_image_num_info = imageToArray(input_faces_path, get_stats=True)
        
        dumpPickleFile(dataX, dataY, labelDict,
                       folderPath=os.path.join(config.path_dict['data_model_path']),
                       picklefileName=pickle_file_name)
        
        dumpCSVFile(person_name_image_num_info,
                    folderPath=os.path.join(config.path_dict['data_model_path']),
                    csvfileName='person_name_image_num_info.csv')
    
    if create_distinct_stratified_batches:
        trainX, trainY, trainLabelDict = getPickleFile(
                config.path_dict['data_model_path'],
                pickle_file_name)
        print(trainX.shape, trainY.shape)
        genDistinctStratifiedBatches(trainX, trainY,
                                     fileName=batch_file_name,
                                     statsFileName='batch_num_image_idx_info.csv'
                                     )
    
    if create_random_stratified_batches:
        trainX, trainY, trainLabelDict = getPickleFile(
                config.path_dict['data_model_path'],
                pickle_file_name)
        print(trainX.shape, trainY.shape)
        genRandomStratifiedBatches(trainX, trainY,
                                   fileName=batch_file_name
                                   )



#####################  TRAIN BEST PARAMS AND TEST  ########################
# USE THE BEST PARAMETER SETTING TO TRAIN THE MODEL
# TRAINING CREATES A CHECKPOINT (TRAINED WEIGHTS) IN THE DISK
# TEST MODULE USES THE CHECKPOINT TO MAKE PREDICTION ON A NEW IMAGE

if train_test_model:
    from config import myNet
    
    test_image_path_arr = [
        '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/output_data_faces/face_snapshot/img2.jpg',
        '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/output_data_faces/face_snapshot/img3.jpg',
        '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/output_data_faces/face_snapshot/img4.jpg'
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

