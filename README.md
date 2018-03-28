
## Deep Face Recognition

### ABOUT THE MODULE
Face recognition softwares are ubiquitous, we use them every day to unlock our cell phones. Deep Learning application have proven to show an increase in performance with increase in data. This application is an attempt to recognize a person given his image. Several ideas are borrowed from the paper <b>FaceNet</b> by Florian Schroff, Dmitry Kalenichenko and James Philbin. The module is developed from scratch using Tensorflow and makes use of transfer learning with Google Net (NN4 small (96x96)) architecture to recognize faces.

### Why Transfer Learning:
From a Deep learning perspective, to perform a classification task we need to have a lot of data to train a good model. Here a lot of Data could be hundreds of thousands. Transfer learning allows us to train a very robust model with a small dataset. It allows us to use a predefined model (trained with millions of data) and augment/finetune on it with respect to out classification problem. It is often suggested to use pretrained models that were trained on tasks similar to our classification problem. Here we use the pre-trained model that was trained on faces. 

### OVERVIEW
The application employs a multi-step approach.
 
1. [Face Detection and Extraction](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/data_transformer/detect_extract_faces.py). The first part is to detect faces given an image. There are many implementation of Face Detection. The implementation we use here is the **Haar classifier**. Haar cascades are obtained online - visit [HERE](https://github.com/opencv/opencv/tree/master/data/haarcascades) for a complete list.   

3. [Face Recognition](https://github.com/Sardhendu/DeepFaceRecognition/tree/master/nn): The second task is to recognize the faces. As discussed above we use the Inception NN4 small architecture. We obtained the pre-trained weights for each layer. 

   * *Training*: Suppose that we have **n** layers. For training we freeze the weights for (1 to n-3) layers (conv1, conv2, conv3, 3a, 3b, 3c, 4a, 4e) and only train weights for the last few layers (5a, 5b, dense). 
   
   * *Cross validation*: 10 Fold cross validation is performed for [Parameter tuning](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/RecognitionTuneParams.ipynb). The model is re-trained, weights are updated using best parameter setting.
   
   * *Classification*: A 128 dimension embedding(feature) is obtained per image using Face Net architecture. A linear SVC (Support Vector Machine) is used to classify a face. 
   
   * *Test*: During test time we first detect and extract all the faces using haar cascade. We then obtained 128 dimension embedding using the updated weight and then classify a face using SVM stored model.

The pre-trained weights for the model can be found [HERE](https://github.com/iwantooxxoox/Keras-OpenFace/tree/master/weights): 

### Get Started with the modules.
Mod 1. [config.py](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/config.py): The config.py file consist all the settings that includes input_data_path, model_path, output_data_path, net_params, trainable_feature_names and etc.

Mod 2: [main.py](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/main.py) (Wraps the complete functionality for data generation, train and test)

  * [Data Generation](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/data_transformer/generate_data.py): This modules calls several functions inside the data_transformer folder. The ultimate goal for this modules is to create 10 fold cross validation batches and dump it in the disk.

  * [Recognition Tune Params](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/RecognitionTuneParams.ipynb): The notebook allows you to play around with different parameters and tune them based on average 10-fold accuracy and output probabilities. Then the best tuning parameter can be chosen for model learning.

  * [Run](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/train_test/run.py): Once the batches are created using **Data Generation** and parameters are decided using **Recognition Tune Params**, This module will trains the model again for the selected model params and stores a checkpoint in the disk. The checkpoints are used by the "Test" module to make predictions.


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


### REFERENCES:
Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering: https://arxiv.org/abs/1503.03832

The pretrained weights were taken from by Victor's github repo: https://github.com/iwantooxxoox/Keras-OpenFace.

The code implementation is also highly inspired from FaceNet github repo: https://github.com/davidsandberg/facenet

The code implementation is also highly inspired from the assignments of Coursera's Deep Learning course on Convolutional Neural Networks. https://www.coursera.org/learn/convolutional-neural-networks