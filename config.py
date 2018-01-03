
import os
global globalDict
path_dict = {}


path_dict['parent_path'] = '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition'
path_dict['data_model_path'] = os.path.join(path_dict['parent_path'] , 'data_models')

path_dict['batchFolderPath'] = os.path.join(path_dict['data_model_path'], 'batch_img_arr')

path_dict['training_encoding_path'] = os.path.join(path_dict['data_model_path'])

path_dict['classification_model_path'] = os.path.join(path_dict['data_model_path'], 'classification_model')

path_dict['inception_nn4small_weights_path'] = "/Users/sam/All-Program/App-DataSet/DeepNeuralNets/Models/FaceNet" \
                                           "_Inception"