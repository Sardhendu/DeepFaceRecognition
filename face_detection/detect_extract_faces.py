from __future__ import division, print_function, absolute_import

import cv2
import numpy as np
import os
from config import path_dict



CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_extract_faces(image_path):
    image=cv2.imread(image_path)
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,
                                          minNeighbors=5,minSize=(25,25),flags=0)
    faces_arr = np.ndarray(shape=())
    for num, (x,y,w,h) in enumerate(faces):
        sub_img=image[y-10:y+h+10,x-10:x+w+10]

        face_dump_path = os.path.join(path_dict['extracted_face_path'],
                                      str(os.path.basename(image_path).split('.')[0]) + str(num) + ".jpg")
        print(face_dump_path)
        cv2.imwrite(face_dump_path, sub_img)
        
        # Place rectangles for the regions
        cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

    cv2.imwrite(os.path.join(path_dict['image_labeled_path'], os.path.basename(image_path)), image)
         

detect_extract_faces(path_dict['image_path'])