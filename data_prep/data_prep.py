from __future__ import division, print_function, absolute_import

import numpy as np
import os, pickle
import logging
from config import path_dict

# parentPath = '/Users/sam/All-Program/App-DataSet/DeepFaceRecognition'
# batchFolderPath = os.path.join(parentPath, 'data_models', 'batch_img_arr')
# training_encoding_path = os.path.join(parentPath, 'data_models')

class DataIO():
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
        print (path_from)
        with open(path_from, "rb") as p:
            data = pickle.load(p)
            dataX = data['dataX']
            dataY = data['dataY']
            labelDict = data['labelDict']
        return dataX, dataY, labelDict


def generateRandomBatches(dataX, dataY, imgPerLabels=6, numBatches=10, fileName=None):
    if isinstance(dataX, list) and len(dataX)==1:
        dataX = dataX[0]
    
    if not isinstance(dataX, np.ndarray):
        raise ValueError('Unhandled type dataX input')
    
    if isinstance(dataY, np.ndarray):
        dataY = dataY.flatten()

    numLabels = len(np.unique(dataY))
    batchSize = imgPerLabels * numLabels
    dataBatchX= np.ndarray(shape=(numBatches, batchSize,
                                     dataX.shape[1], dataX.shape[2], dataX.shape[3]))
    dataBatchY= np.ndarray(shape=(numBatches, batchSize))
    for batch_num in np.arange(numBatches):
        logging.info('Running for batch %s ', str(batch_num))
        batchX = np.ndarray(shape=(batchSize, dataX.shape[1], dataX.shape[2], dataX.shape[3]))
        batchY = np.zeros(batchSize)
        for iter, labels in enumerate(np.unique(dataY)):
            logging.info('Running for label %s ', str(labels))
            label_idx = np.where(dataY == labels)[0]
            np.random.shuffle(label_idx)
            i = iter*imgPerLabels
            j = (iter+1)*imgPerLabels
            batchX[i:j,:] = dataX[label_idx[0:imgPerLabels]]
            batchY[i:j] = dataY[label_idx[0:imgPerLabels]]
        dataBatchX[batch_num, :] = batchX
        dataBatchY[batch_num, :] = batchY
    logging.info('The Data batches dumped has shape: ', str(dataBatchX.shape))
    logging.info('The Label batch dumped has shape: ', str(dataBatchY.shape))
    
    if fileName:
        DataIO.dumpPickleFile(dataX=dataBatchX,
                              dataY=dataBatchY,
                              labelDict=None,
                              folderPath=path_dict['batchFolderPath'],
                              picklefileName=fileName)

    
    
debugg = False

if debugg:
    trainX, trainY, trainLabelDict = DataIO.getPickleFile(path_dict['training_encoding_path'],
                                                                 'training_imgarr.pickle')
    verX, verY, verLabelDict = DataIO.getPickleFile(path_dict['training_encoding_path'],
                                                           'verification_imgarr.pickle')
    print(trainX.shape, trainY.shape)
    print(verX.shape, verY.shape)
    generateRandomBatches(trainX, trainY, imgPerLabels=6, numBatches=10,
                          fileName='random_batches.pickle')