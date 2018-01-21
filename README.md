
## Deep Face Recognition

### ABOUT THE MODULE
This Application is an inspiration from Coursera's lecture (deeplearning.ai) on Face Recognition and it implements ideas from "FaceNet" by Florian Schroff, Dmitry Kalenichenko and James Philbin. 

The module is developed from scratch using Tensorflow. The application makes the use of Inception Net NN4 small (96x96) architecture.


### OVERVIEW
The application works in two simple ways.

1. **Face Detection and Extraction**. There are many implementation of Face Detection. The implementation we use here is the Haar classifier. Haar cascades are obtained online and used here. Visit https://github.com/opencv/opencv/tree/master/data/haarcascades for a complete list.
  
2. **Face Recognition**: As discussed above we use the Inception NN4 small architecture. We obtained the pre-trained weights for each layer. 

   * *Training*: For training we freeze the weights for (1 to n-3) layers (conv1, conv2, conv3, 3a, 3b, 3c, 4a, 4e) and only train weights for the last few layers (5a, 5b, dense). 
   
   * *Cross validation*: 10 Fold cross validation is performed for parameter tuning. The model is re-trained, weights are updated using best parameter setting.
   
   * *Classification*: A 128 dimension embedding(feature) is obtained per image using Face Net architecture. A linear SVC (Support Vector Machine) is use to classify a face. 
   
   * *Test*: During test time we first detect and extract all the faces using haar cascade. We then obtained 128 dimension embedding using the updated weight and then classify a face using SVM stored model.

### Training Images
