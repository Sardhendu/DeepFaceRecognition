
import os
global path_dict
global myNet
global seed_arr
global weight_seed_idx

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

myNet['img_per_label'] = 6
myNet['num_labels'] = 3
myNet['learning_rate'] = 0.0001



#############  ACT IN SEED
seed_arr = [213,436,754,991,302,992,223,645,724,944,232,123,321,
            909,784,239,337,888,666, 400,912,255,983,902,846,345,
            854,989,291,486,444,101,202,304,505,607,707,808,905, 900,
            774,553,292,394,874,445,191,161,141,272]




weight_seed_idx = 0
triplet_seed_idx = 0
preprocess_seed_idx = 0