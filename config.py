
import os
global path_dict
global myNet
global vars
global seed_arr
global weight_seed_idx
global finetune_layer_scope_names

path_dict = {}


path_dict['parent_path'] = '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition'
path_dict['data_model_path'] = os.path.join(path_dict['parent_path'] , 'data_models')

path_dict['batchFolderPath'] = os.path.join(path_dict['data_model_path'], 'batch_img_arr')

path_dict['training_encoding_path'] = os.path.join(path_dict['data_model_path'])

path_dict['classification_model_path'] = os.path.join(path_dict['data_model_path'], 'classification_model')

path_dict['inception_nn4small_weights_path'] = "/Users/sam/All-Program/App-DataSet/DeepNeuralNets/Models/FaceNet" \
                                           "_Inception"

path_dict['checkpoint_path'] = os.path.join(path_dict['data_model_path'], 'saver_checkpoints')
path_dict['summary_path'] = os.path.join(path_dict['data_model_path'], 'summary')


myNet = {}


myNet['image_shape'] = [96, 96, 3]
myNet['use_checkpoint'] = False

myNet['triplet_selection_alpha'] = 0.01
myNet['triplet_loss_penalty'] = 0.2

myNet['img_per_label'] = 10
myNet['num_labels'] = 3
myNet['learning_rate'] = 0.001
myNet['learning_rate_decay_rate'] = 0.95
myNet['batch_norm_decay'] = 0.9


vars = {}
vars['numBatches'] = 10
vars['numImgsPerLabels'] = 100
vars['batchSize'] = 30
vars['trainSize'] = vars['numImgsPerLabels'] * myNet['num_labels'] - vars['batchSize']

#############  ACT IN SEED
seed_arr = [213,436,754,991,302,992,223,645,724,944,232,123,321,
            909,784,239,337,888,666, 400,912,255,983,902,846,345,
            854,989,291,486,444,101,202,304,505,607,707,808,905, 900,
            774,553,292,394,874,445,191,161,141,272]




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
    
