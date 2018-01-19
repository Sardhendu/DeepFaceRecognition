
import os
import pickle
import logging
import numpy as np

def dumpPickleFile(dataX, dataY, labelDict=None, folderPath=None, picklefileName=None, getStats=None):
    if not folderPath or not picklefileName:
        raise ValueError('You should provide a folder path and pickle file name to dump your file')
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    path_to_dump = os.path.join(folderPath, picklefileName)
    
    logging.info('DATA FORMATTER: Dumping the pickle file %s to disk, dataX shape = %s, dataY shape = %s',
                 str(picklefileName),
                 str(dataX.shape),
                 str(dataY.shape))
    if getStats:
        print('The shape of input data (X) is: ', np.array(dataX).shape)
        print('The shape of input data (Y) is: ', np.array(dataY).shape)
        print('Unique labels in dataY is: ', np.unique(dataY))
        print('Label dict: ', labelDict)
    
    with open(path_to_dump, 'wb') as f:
        fullData = {
            'dataX': dataX,
            'dataY': dataY,
            'labelDict': labelDict
        }
        pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)

def dumpCSVFile(dataDF, folderPath, csvfileName, getStats=None):
    path = os.path.join(folderPath, csvfileName)
    dataDF.to_csv(path, index=None)

def getPickleFile(folderPath, picklefileName, getStats=None):
    path_from = os.path.join(folderPath, picklefileName)
    with open(path_from, "rb") as p:
        data = pickle.load(p)
        dataX = data['dataX']
        dataY = data['dataY']
        labelDict = data['labelDict']
    
    logging.info('DATA FORMATTER: Retrieved the pickle file %s from disk, dataX shape = %s, dataY shape = %s',
                 str(picklefileName),
                 str(dataX.shape),
                 str(dataY.shape))
    if getStats:
        print('The shape of input data (X) is: ', np.array(dataX).shape)
        print('The shape of input data (Y) is: ', np.array(dataY).shape)
        print('Unique labels in dataY is: ', np.unique(dataY))
        print('Label dict: ', labelDict)
    return dataX, dataY, labelDict



