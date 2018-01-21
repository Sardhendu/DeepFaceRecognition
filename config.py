
import os
global path_dict
global myNet
global vars
global seed_arr
global weight_seed_idx
global finetune_layer_scope_names


path_dict = {}

# /Users/sam/All-Program/App-DataSet/DeepFaceRecognition/summary

path_dict['parent_path'] = '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition'


# PRE-OBTAINED HAAR CASCADES FOR FACE DETECTION
path_dict['haar_cascade'] = os.path.join(path_dict['parent_path'], "Face_cascade.xml")

# PRE-TRAINED INCEPTION WEIGHT DIRECTORY
path_dict['inception_nn4small_weights_path'] = "/Users/sam/All-Program/App-DataSet/DeepNeuralNets/Models/FaceNet" \
                                           "_Inception"

# INPUT DATA PATHS
path_dict['face_snapshot_path'] = os.path.join(path_dict['parent_path'], 'input_data_faces', 'face_snapshot')
path_dict['face_snapshot_resized_path'] = os.path.join(path_dict['parent_path'], 'input_data_faces', 'face_snapshot_resized')
path_dict['face_extracted_path'] = os.path.join(path_dict['parent_path'], 'input_data_faces', 'face_extracted')
path_dict['face_detection_path'] = os.path.join(path_dict['parent_path'], 'input_data_faces', 'face_detection')


# OUTPUT DATA PATHS
path_dict['face_snapshot_test_path'] = os.path.join(path_dict['parent_path'], 'output_data_faces', 'face_snapshot')
path_dict['face_extracted_test_path'] = os.path.join(path_dict['parent_path'], 'output_data_faces', 'face_extracted')
path_dict['face_detection_test_path'] = os.path.join(path_dict['parent_path'], 'output_data_faces', 'face_detection')
path_dict['face_detection_labeled_test_path'] = os.path.join(path_dict['parent_path'], 'output_data_faces', 'face_detection_labeled')


# MODEL, PARAMS AND ANALYSIS DATA PATH
path_dict['analysis_path'] = os.path.join(path_dict['parent_path'], 'analysis')
path_dict['data_model_path'] = os.path.join(path_dict['parent_path'] , 'data_models')
path_dict['batchFolderPath'] = os.path.join(path_dict['data_model_path'], 'batch_img_arr')
path_dict['training_encoding_path'] = os.path.join(path_dict['data_model_path'])
path_dict['classification_model_path'] = os.path.join(path_dict['data_model_path'], 'classification_model')
path_dict['checkpoint_path'] = os.path.join(path_dict['data_model_path'], 'saver_checkpoints')
path_dict['summary_path'] = os.path.join(path_dict['data_model_path'], 'summary')
path_dict['cv_pred_analysis_path'] = os.path.join(path_dict['analysis_path'], 'cv_pred_analysis')

# path_dict['image_path'] = '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/extras/full_images/now_you_see_me.jpg'
# path_dict['extracted_face_path'] = '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/extras/faces_extracted/'
#
# path_dict['image_labeled_path'] = "/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/extras/full_images_labeled"






#####################   NET PARAMETERS
myNet = {}
myNet['image_shape'] = [96, 96, 3]

myNet['triplet_selection_alpha'] = 0.09  # The lesser this value is, the more hard negative we select. Try a range between 0.01 to 0.05
myNet['triplet_loss_penalty'] = 0.1 # The larger this value is, the more penalty we induce. For example, That is if anc_vs_pos = 0.3 and anc_vs_neg = 0.9 then 0.3-0.9+0.2 = -0.4 then -0.4<0. So the loss is 0. On other hand, if anc_vs_pos = 0.6 and anc_vs_neg = 0.7 then 0.6-0.7+0.2 = 0.1 then 0.1>0. So the loss is 0.1

myNet['img_per_label'] = 10
myNet['num_labels'] = 3
myNet['keep_prob'] = 0.7
myNet['learning_rate'] = 0.001
myNet['learning_rate_decay_rate'] = 0.95
myNet['batch_norm_decay'] = 0.9

#####################   BATCH PARAMETERS

vars = {}
vars['numBatches'] = 10
vars['numImgsPerLabels'] = 100
vars['batchSize'] = 30
vars['trainSize'] = vars['numImgsPerLabels'] * myNet['num_labels'] - vars['batchSize']

####################    PREPROCESSING PARAMETERS
pp_vars = {}
pp_vars['standardise'] = True
pp_vars['rand_brightness'] = True
pp_vars['rand_contrast'] = True
pp_vars['rand_rotate'] = False
pp_vars['rand_flip'] = True
pp_vars['rand_crop'] = False


#############  ACT IN SEED
seed_arr = [553, 292, 394, 874, 445, 191, 161, 141, 213,436,754,991,302,992,223,645,724,944,232,
            123,321,
            909,784,239,337,888,666, 400,912,255,983,902,846,345,
            854,989,291,486,444,101,202,304,505,607,707,808,905, 900,
            774,272]




weight_seed_idx = 0
triplet_seed_idx = 0
preprocess_seed_idx = 0



# The fine tune scope names are very important. They should exactly match the names provided in
# tensorflow tf.GraphKeys.GLOBAL_VARIABLES. Because these names are used as scope names for variable while training,
# during crossvalidation or test these scope name are used to retrieve the last updated value
# of the parameters, which is again used to update the parameters.

# Moreover it is good to ensure that this list contains all the scope involved in fine tuning
# If not then the output would be inaccurate.


finetune_variables = [
    'inception_5a_1x1_conv/w:0', 'inception_5a_1x1_conv/b:0',
    'inception_5a_1x1_bn/w:0', 'inception_5a_1x1_bn/b:0','inception_5a_1x1_bn/m:0', 'inception_5a_1x1_bn/v:0',
    'inception_5a_3x3_conv1/w:0', 'inception_5a_3x3_conv1/b:0',
    'inception_5a_3x3_conv2/w:0', 'inception_5a_3x3_conv2/b:0',
    'inception_5a_3x3_bn1/w:0', 'inception_5a_3x3_bn1/b:0', 'inception_5a_3x3_bn1/m:0', 'inception_5a_3x3_bn1/v:0',
    'inception_5a_3x3_bn2/w:0', 'inception_5a_3x3_bn2/b:0', 'inception_5a_3x3_bn2/m:0', 'inception_5a_3x3_bn2/v:0',
    'inception_5a_pool_conv/w:0', 'inception_5a_pool_conv/b:0',
    'inception_5a_pool_bn/w:0', 'inception_5a_pool_bn/b:0', 'inception_5a_pool_bn/m:0', 'inception_5a_pool_bn/v:0',
    
    'inception_5b_1x1_conv/w:0', 'inception_5b_1x1_conv/b:0',
    'inception_5b_1x1_bn/w:0', 'inception_5b_1x1_bn/b:0','inception_5b_1x1_bn/m:0', 'inception_5b_1x1_bn/v:0',
    'inception_5b_3x3_conv1/w:0', 'inception_5b_3x3_conv1/b:0',
    'inception_5b_3x3_conv2/w:0', 'inception_5b_3x3_conv2/b:0',
    'inception_5b_3x3_bn1/w:0', 'inception_5b_3x3_bn1/b:0', 'inception_5b_3x3_bn1/m:0', 'inception_5b_3x3_bn1/v:0',
    'inception_5b_3x3_bn2/w:0', 'inception_5b_3x3_bn2/b:0', 'inception_5b_3x3_bn2/m:0', 'inception_5b_3x3_bn2/v:0',
    'inception_5b_pool_conv/w:0', 'inception_5b_pool_conv/b:0',
    'inception_5b_pool_bn/w:0', 'inception_5b_pool_bn/b:0', 'inception_5b_pool_bn/m:0', 'inception_5b_pool_bn/v:0',
    
    'dense/w:0', 'dense/b:0']
    
