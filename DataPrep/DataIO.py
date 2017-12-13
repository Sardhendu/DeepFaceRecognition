import os
import numpy as np
import pickle
from scipy import ndimage, misc
from skimage import transform, io, img_as_uint

parent_path = "/Users/sam/All-Program/App-DataSet/DeepFaceRecognition/"


class DataFormatter():
    def __init__(self, parentPath, whichData):
        self.parentPath = parentPath
        
        if whichData == "training":
            self.original_image_path = os.path.join(self.parentPath, 'original', 'training')
            self.resized_path = os.path.join(self.parentPath, 'resized','training')
        elif whichData == "verification":
            self.original_image_path = os.path.join(self.parentPath, 'original', 'verification')
            self.resized_path = os.path.join(self.parentPath, 'resized','verification')
            
        print ('Getting images from : ', self.original_image_path)
        print('Dumping resized images to : ', self.resized_path)
    
    def createResizedData(self):
        folderPath = self.original_image_path
        people = [folder for folder in os.listdir(folderPath) if len(folder.split(".")) == 1]
        
        for name in people:
            outPersonPath = os.path.join(self.resized_path, name)
            
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
        if not os.path.exists(self.resized_path):
            raise ValueError('You should run the resize image dump before running imageToArray')
        
        person_names = [person_name for person_name in os.listdir(self.resized_path)
                        if person_name != ".DS_Store"]
        dataX = []
        dataY = []
        labelDict = {}
        for label, person_name in enumerate(person_names):
            photo_path = os.path.join(self.resized_path, person_name)
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
    def dumpPickleFile(dataX, dataY, labelDict=None, folderPath=None, picklefileName=None):
        if not folderPath or not picklefileName:
            raise ValueError('You should provide a folder path and pickle file name to dump your file')
        
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        
        path_to_dump = os.path.join(folderPath, picklefileName)
        
        with open(path_to_dump, 'wb') as f:
            fullData = {
                'dataX': dataX,
                'dataY': dataY,
                'labelDict': labelDict
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)
         
    @staticmethod
    def getPickleFile(folderPath, picklefileName):
        path_from = os.path.join(folderPath, picklefileName)
        with open(path_from, "rb") as p:
            data = pickle.load(p)
            dataX = data['dataX']
            dataY = data['dataY']
            labelDict = data['labelDict']
        return dataX[0], dataY, labelDict


training = False
verification = False

if training:
    objDP = DataFormatter(parent_path, 'training')
    objDP.createResizedData()
    dataX, dataY, labelDict = objDP.imageToArray()
    DataFormatter.dumpPickleFile(dataX, dataY, labelDict,
                               folderPath=os.path.join(parent_path, 'data_models'),
                               picklefileName='training_imgarr.pickle')
if verification:
    objDP = DataFormatter(parent_path, 'verification')
    objDP.createResizedData()
    dataX, dataY, labelDict = objDP.imageToArray()
    DataFormatter.dumpPickleFile(dataX, dataY, labelDict,
                               folderPath=os.path.join(parent_path, 'data_models'),
                               picklefileName='verification_imgarr.pickle')
    
 

######################################  OBTAIN TRIPLETS
from itertools import combinations
np.random.seed(327)
embedding_per_batch = np.random.rand(40, 2)
batch_size, num_embeddings = embedding_per_batch.shape
img_per_label = 10
num_labels = int(batch_size/img_per_label)
idx_arr = np.arange(batch_size)
alpha = 0.01

def get_triplets():
    batch_tripet_idx = []
    for i in np.arange(num_labels):
        pos_idxs = np.arange(i*img_per_label,i*img_per_label+img_per_label)
        neg_idxs = np.setdiff1d(idx_arr, pos_idxs)
        # print(pos_idxs)
        # print('')
        # print(neg_idxs)
        compare_point = -1  # used to avoid redundancy in calculating SSE between anchor and all negative
        # Get all combination of Anchor and positive
        # print ('######################################')
        for anc_idx, pos_idx in combinations(pos_idxs, 2):
            if anc_idx != compare_point:
                compare_point += 1
                # Get the sum of squared distance between anchor and positive
                anc_VS_neg_ndarr = np.sum(np.square(
                        embedding_per_batch[anc_idx] - embedding_per_batch[neg_idxs]), 1
                )
                # print (anc_VS_neg_ndarr)
                
            # Get the sum of squared distance between anchor and positive
            anc_VS_pos = np.sum(
                    np.square(embedding_per_batch[anc_idx] - embedding_per_batch[pos_idx]))
            # print ('anc_VS_pos anc_VS_pos', anc_VS_pos)
    
            hard_neg_idx = np.where(anc_VS_neg_ndarr - anc_VS_pos < alpha)[0]
            
            
            # Randomly sample 1 record from the hard negative idx, and create a triplet
            if len(hard_neg_idx)>0:
                # print('hard_neg_idx hard_neg_idx ', hard_neg_idx)
                np.random.shuffle(hard_neg_idx)
                rnd_idx = hard_neg_idx[0]
                # print ('neg_idx neg_idx ', neg_idxs[rnd_idx])
                
                # Get triplet indexes in order to work on offline mode
                batch_tripet_idx.append([anc_idx, pos_idx, neg_idxs[rnd_idx]])
            # print (anc_idx,pos_idx)
        # print ('')
            # print (break_point - cnt)
    # print('TRIPLET INDEX ', batch_tripet_idx)
    
    return batch_tripet_idx

batch_tripet_idx = get_triplets()

print (batch_tripet_idx)