
import os
global globalDict
path_dict = {}


path_dict['parentPath'] = '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition'

path_dict['batchFolderPath'] = os.path.join(path_dict['parentPath'], 'data_models', 'batch_img_arr')
path_dict['training_encoding_path'] = os.path.join(path_dict['parentPath'], 'data_models')


path_dict['classification_model_path'] = os.path.join(path_dict['parentPath'], 'data_models', 'classification_model')

path_dict['inception_nn4small_weights_path'] = "/Users/sam/All-Program/App-DataSet/DeepNeuralNets/Models/FaceNet" \
                                           "_Inception"