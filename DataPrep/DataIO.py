import os
import numpy as np
import pickle
from scipy import ndimage, misc
from skimage import transform, io, img_as_uint

parent_path = "/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/"


class DataFormatter():
    def __init__(self, parentPath):
        self.parentPath = parentPath
        self.original_image_path = os.path.join(self.parentPath, 'original')
        self.resized_image_path = os.path.join(self.parentPath, 'resized')
    
    def createResizedData(self):
        folderPath = self.original_image_path
        people = [folder for folder in os.listdir(folderPath) if len(folder.split(".")) == 1]
        
        for name in people:
            outPersonPath = os.path.join(self.resized_image_path, name)
            
            if not os.path.exists(outPersonPath):
                os.makedirs(outPersonPath)
                
            personPath = os.path.join(folderPath, name)
                    
            imagePathList = [os.path.join(personPath, images) for images in os.listdir(personPath)
                             if np.array(images.split("."))[-1] == 'jpg'
                             or np.array(images.split('.'))[-1] == "jpeg"]
            
            for num, imagePath in enumerate(imagePathList):
                # print(imagePath)
                image = ndimage.imread(imagePath, mode='RGB')
            
                imageResized = misc.imresize(image, (96, 96))
                io.imsave(os.path.join(outPersonPath, '%s.jpg' % str(num)), img_as_uint(imageResized))
                
                
    def imageToArray(self):
        if not os.path.exists(self.resized_image_path):
            raise ValueError('You should run the resize image dump before running imageToArray')
        
        person_names = [person_name for person_name in os.listdir(self.resized_image_path)
                        if person_name != ".DS_Store"]
        dataX = []
        dataY = []
        labelDict = {}
        for label, person_name in enumerate(person_names):
            photo_path = os.path.join(self.resized_image_path, person_name)
            person_pics = [os.path.join(photo_path, pics) for pics in os.listdir(photo_path) if
                           pics.split('.')[1] == "jpg" or pics.split('.')[1] == "jpeg"]
            labelDict[str(label)] = person_name
            per_person_img_arr = np.ndarray((len(person_pics), 96, 96, 3))
            per_person_labels = np.tile(label, len(person_pics)).reshape(-1, 1)
            for img_num, pic_path in enumerate(person_pics):
                image = ndimage.imread(pic_path, mode='RGB')
                per_person_img_arr[img_num, :] = image
            if label == 0:
                dataX = per_person_img_arr
                dataY = per_person_labels
            else:
                dataX = np.vstack((dataX, per_person_img_arr))
                dataY = np.vstack((dataY, per_person_labels))
        return dataX, dataY, labelDict
    
    @staticmethod
    def dumpImageArr(self, dataX, dataY, labelDict=None, path_to_dump=None):
        if not path_to_dump:
            path_to_dump = os.path.join(self.parentPath, 'full_data_ndarr.pickle')

        with open(path_to_dump, 'wb') as f:
            fullData = {
                'dataX': dataX,
                'dataY': dataY,
                'labelDict': labelDict
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)


debugg = False
if debugg:
    objDP = DataFormatter(parent_path)
    objDP.createResizedData()
    dataX, dataY, labelDict = objDP.imageToArray()
    objDP.dumpImageArr(dataX, dataY, labelDict)