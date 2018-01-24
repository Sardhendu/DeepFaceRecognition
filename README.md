
## Deep Face Recognition

### ABOUT THE MODULE
This Application implements ideas from "FaceNet" by Florian Schroff, Dmitry Kalenichenko and James Philbin. https://arxiv.org/abs/1503.03832. The module is developed from scratch using Tensorflow. The application makes use of the Inception Net NN4 small (96x96) architecture.


### OVERVIEW
The application works in two simple ways.

1. **Face Detection and Extraction**. There are many implementation of Face Detection. The implementation we use here is the Haar classifier. Haar cascades are obtained online and used here. Visit https://github.com/opencv/opencv/tree/master/data/haarcascades for a complete list.
  
2. **Face Recognition**: As discussed above we use the Inception NN4 small architecture. We obtained the pre-trained weights for each layer. 

   * *Training*: For training we freeze the weights for (1 to n-3) layers (conv1, conv2, conv3, 3a, 3b, 3c, 4a, 4e) and only train weights for the last few layers (5a, 5b, dense). 
   
   * *Cross validation*: 10 Fold cross validation is performed for parameter tuning. The model is re-trained, weights are updated using best parameter setting.
   
   * *Classification*: A 128 dimension embedding(feature) is obtained per image using Face Net architecture. A linear SVC (Support Vector Machine) is used to classify a face. 
   
   * *Test*: During test time we first detect and extract all the faces using haar cascade. We then obtained 128 dimension embedding using the updated weight and then classify a face using SVM stored model.

### Training Images
<img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/sample_training_image/37.jpg" width="100" height="100">
<img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/sample_training_image/4.jpg" width="100" height="100">
<img src="https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/sample_training_image/5.jpg" width="100" height="100">



### Test Image
![test image size](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_snapshot/img2.jpg){:height="50%" width="50%"}
![test image size](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_snapshot/img4.jpg){:height="50%" width="50%"}

### Test Image Face Detected
![test image size](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_detection/img2.jpg){:height="50%" width="50%"}
![test image size](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_detection/img4.jpg){:height="50%" width="50%"}

### Test Image Face Extracted
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_extracted/img2_0.jpg)
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_extracted/img2_1.jpg)
![alt text](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_extracted/img4_0.jpg)

### Test Image Labeled
![test image size](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_detection_labeled/img2.jpg){:height="50%" width="50%"}
![test image size](https://github.com/Sardhendu/DeepFaceRecognition/blob/master/images/face_detection_labeled/img4.jpg){:height="50%" width="50%"}