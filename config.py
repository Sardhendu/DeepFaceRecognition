
import os
global path_dict
global myNet

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