from __future__ import division, print_function, absolute_import

import cv2
import numpy as np
import os
from scipy import  misc

from config import path_dict, myNet

FACE_CASCADE=cv2.CascadeClassifier(path_dict['haar_cascade'])

def resize_image(image, resize_shape):
    if image.shape[0] < resize_shape[0] or image.shape[1] < resize_shape[1]:
        return []
    else:
        imageResized = misc.imresize(image, myNet['image_shape'])
        imageResized = np.array(imageResized)

        return imageResized
    

def detect_extract_faces(image_path):
    image=cv2.imread(image_path)
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,
                                          minNeighbors=5,minSize=(25,25),flags=0)
    faces_arr = []
    for num, (x,y,w,h) in enumerate(faces):
        sub_img = image[y-10:y+h+10,x-10:x+w+10]
        print(sub_img.shape)
        sub_img = resize_image(sub_img, resize_shape=myNet['image_shape'])
        
        if len(sub_img) > 0:
            print (sub_img.shape)
            face_dump_path = os.path.join(path_dict['extracted_face_path'],
                                          str(os.path.basename(image_path).split('.')[0]) + str(num) + ".jpg")
            print(face_dump_path)
            faces_arr.append(sub_img)
            cv2.imwrite(face_dump_path, sub_img)
            
        # Place rectangles for the regions
        cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

    cv2.imwrite(os.path.join(path_dict['image_labeled_path'], os.path.basename(image_path)), image)
    return np.array(faces_arr)
   
   
debugg = False
if debugg:
    full_image_path = '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/extras/full_images/img1.jpg'
    test_faces = detect_extract_faces(full_image_path)
    print (test_faces.shape)
# detect_extract_faces(path_dict['image_path'])