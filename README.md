
## Deep Face Recognition

### ABOUT THE MODULE
This Application implements ideas from "FaceNet" by Florian Schroff, Dmitry Kalenichenko and James Philbin. https://arxiv.org/abs/1503.03832. The module is developed from scratch using Tensorflow. The application makes use of transfer learning with Google Net (NN4 small (96x96)) architecture.


### OVERVIEW
The application works in two simple ways.

1. **Face Detection and Extraction**. There are many implementation of Face Detection. The implementation we use here is the Haar classifier. Haar cascades are obtained online and used here. Visit https://github.com/opencv/opencv/tree/master/data/haarcascades for a complete list.
  
2. **Face Recognition**: As discussed above we use the Inception NN4 small architecture. We obtained the pre-trained weights for each layer. 

   * *Training*: For training we freeze the weights for (1 to n-3) layers (conv1, conv2, conv3, 3a, 3b, 3c, 4a, 4e) and only train weights for the last few layers (5a, 5b, dense). 
   
   * *Cross validation*: 10 Fold cross validation is performed for parameter tuning. The model is re-trained, weights are updated using best parameter setting.
   
   * *Classification*: A 128 dimension embedding(feature) is obtained per image using Face Net architecture. A linear SVC (Support Vector Machine) is used to classify a face. 
   
   * *Test*: During test time we first detect and extract all the faces using haar cascade. We then obtained 128 dimension embedding using the updated weight and then classify a face using SVM stored model.

### Get Started.
Mod 1. config.py: The config.py file consist all the settings that includes input_data_path, model_path, output_data_path, net_params, trainable_feature_names and etc.

Mod 2: main.py (Wraps the complete functionality for data generation, train and test)

  * Func1: generate_data.py: (../data_transformer/generate_data.py) This modules calls several functions inside the data_transformer folder. The ultimate goal for this modules is to create 10 fold cross validation batches and dump it in the respected path for the neural network model to pick up.

  * Func 2. RecognitionTuneParams.ipynb: The notebook allows you to play arround with different parameters and tune them based on average 10-fold accuracy and output probabilities.

  * Func 3. run.py: (../train_test/run.py): Once the batches are created using Mod 2 and parameters are decided using Mod 3, run.py trains the model again for the selected model params and stores a checkpoint in the disk. The checkpoints are used by the "Test" module to make predictions.


### Training Images
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/sample_training_image/37.jpg)
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/sample_training_image/4.jpg)
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/sample_training_image/5.jpg)


### Test Image
<img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_snapshot/img2.jpg" width="400" 
height="300"> <img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_snapshot/img4.jpg" 
width="400" height="300">

### Test Image Face Detected
<img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_detection/img2.jpg" width="400" 
height="300"> <img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_detection/img4.jpg" 
width="400" height="300">

### Test Image Face Extracted
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_extracted/img2_0.jpg)
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_extracted/img2_1.jpg)
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_extracted/img4_0.jpg)

### Test Image Labeled
<img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_detection_labeled/img2.jpg" 
width="400" height="300"> <img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_detection_labeled/img4.jpg" width="400" height="300">