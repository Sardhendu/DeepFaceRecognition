import os
import numpy as np
import pandas as pd
from scipy import ndimage
from config import path_dict, myNet, vars
from data_transformer.data_io import  dumpPickleFile
import logging


def imageToArray(input_faces_path, get_stats=None):
    if not os.path.exists(input_faces_path):
        raise ValueError('You should run the resize image dump before running imageToArray')
    
    person_names = [person_name for person_name in os.listdir(input_faces_path)
                    if person_name != ".DS_Store"]
    dataX = []
    dataY = []
    labelDict = {}
    pic_filename_arr = []
    person_name_arr = []
    label_arr = []
    for label, person_name in enumerate(person_names):
        photo_path = os.path.join(input_faces_path, person_name)
        person_pics = [pics.split('.')[0]
                       for pics in os.listdir(photo_path)
                       if pics.split('.')[1] == "jpg" or pics.split('.')[1] == "jpeg"]
        # Extra line of code, just to sort properly
        person_pics = [os.path.join(photo_path, str(pic) + '.jpg')
                       for pic in sorted(list(map(int, person_pics)))]
        
        labelDict[str(label)] = person_name
        per_person_img_arr = np.ndarray(
                (len(person_pics), myNet['image_shape'][0], myNet['image_shape'][1], myNet['image_shape'][2]))
        per_person_labels = np.tile(label, len(person_pics)).reshape(-1, 1)
        
        for img_num, pic_path in enumerate(person_pics):
            pic_filename_arr += [os.path.basename(pic_path).split('.')[0]]
            image = ndimage.imread(pic_path, mode='RGB')
            per_person_img_arr[img_num, :] = image
        
        person_name_arr += [person_name] * len(person_pics)
        label_arr += [str(label)] * len(person_pics)
        
        if label == 0:
            dataX = per_person_img_arr
            dataY = per_person_labels
        else:
            dataX = np.vstack((dataX, per_person_img_arr))
            dataY = np.vstack((dataY, per_person_labels))
    
    if get_stats:
        person_name_image_num_info = np.column_stack((
            np.array(label_arr).reshape(-1, 1),
            np.array(person_name_arr).reshape(-1, 1),
            np.array(pic_filename_arr).reshape(-1, 1)
        ))
        person_name_image_num_info = pd.DataFrame(
                person_name_image_num_info,
                columns=['image_label',
                         'person_name',
                         'file_name'])
        person_name_image_num_info = person_name_image_num_info.reset_index()
        
        return dataX, dataY, labelDict, person_name_image_num_info
    else:
        return dataX, dataY, labelDict


def genRandomStratifiedBatches(dataX, dataY, fileName=None):
    if isinstance(dataX, list) and len(dataX) == 1:
        dataX = dataX[0]
    
    if not isinstance(dataX, np.ndarray):
        raise ValueError('Unhandled type dataX input')
    
    if isinstance(dataY, np.ndarray):
        dataY = dataY.flatten()
    
    img_per_lbl_per_btch = int(np.round(vars['numImgsPerLabels'] / vars['numBatches']))
    
    numLabels = len(np.unique(dataY))
    batchSize = img_per_lbl_per_btch * numLabels
    
    dataBatchX = np.ndarray(shape=(vars['numBatches'], batchSize,
                                   dataX.shape[1], dataX.shape[2], dataX.shape[3]))
    dataBatchY = np.ndarray(shape=(vars['numBatches'], batchSize))
    
    for batch_num in np.arange(vars['numBatches']):
        logging.info('Running for batch %s ', str(batch_num))
        batchX = np.ndarray(shape=(batchSize, dataX.shape[1], dataX.shape[2], dataX.shape[3]))
        batchY = np.zeros(batchSize)
        for iter, labels in enumerate(np.unique(dataY)):
            logging.info('Running for label %s ', str(labels))
            label_idx = np.where(dataY == labels)[0]
            np.random.shuffle(label_idx)
            i = iter * img_per_lbl_per_btch
            j = (iter + 1) * img_per_lbl_per_btch
            batchX[i:j, :] = dataX[label_idx[0:img_per_lbl_per_btch]]
            batchY[i:j] = dataY[label_idx[0:img_per_lbl_per_btch]]
        dataBatchX[batch_num, :] = batchX
        dataBatchY[batch_num, :] = batchY
    
    if fileName:
        if not os.path.exists(path_dict['batchFolderPath']):
            os.makedirs(path_dict['batchFolderPath'])
        logging.info('The Data batches dumped has shape: %s', str(dataBatchX.shape))
        logging.info('The Label batch dumped has shape: %s', str(dataBatchY.shape))
        dumpPickleFile(dataX=dataBatchX,
                              dataY=dataBatchY,
                              labelDict=None,
                              folderPath=path_dict['batchFolderPath'],
                              picklefileName=fileName)


def genDistinctStratifiedBatches(dataX, dataY, fileName=None, statsFileName=None):
    if isinstance(dataX, list) and len(dataX) == 1:
        dataX = dataX[0]
    
    if not isinstance(dataX, np.ndarray):
        raise ValueError('Unhandled type dataX input')
    
    if isinstance(dataY, np.ndarray):
        dataY = dataY.flatten()
    
    img_per_lbl_per_btch = int(np.round(vars['numImgsPerLabels'] / vars['numBatches']))
    
    numLabels = len(np.unique(dataY))
    batchSize = img_per_lbl_per_btch * numLabels
    
    dataBatchX = np.ndarray(shape=(vars['numBatches'], batchSize,
                                   dataX.shape[1], dataX.shape[2], dataX.shape[3]))
    dataBatchY = np.ndarray(shape=(vars['numBatches'], batchSize))
    
    label_arr = []
    batch_num_arr = []
    image_idx_arr = []
    print(vars['numBatches'])
    for batch_num in np.arange(vars['numBatches']):
        logging.info('Running for batch %s ', str(batch_num))
        batchX = np.ndarray(shape=(batchSize, dataX.shape[1], dataX.shape[2], dataX.shape[3]))
        batchY = np.zeros(batchSize)
        a = batch_num * img_per_lbl_per_btch
        b = (batch_num + 1) * img_per_lbl_per_btch
        # print ('a,b ', a,b)
        for iter, labels in enumerate(np.unique(dataY)):
            logging.info('Running for label %s ', str(labels))
            label_idx = np.where(dataY == labels)[0]
            np.random.seed(1782)
            np.random.shuffle(label_idx)
            # print ('Running for label :', labels)
            # print (label_idx)
            i = iter * img_per_lbl_per_btch
            j = (iter + 1) * img_per_lbl_per_btch
            # print('i, j ',i, j)
            batchX[i:j, :] = dataX[label_idx[a:b]]
            batchY[i:j] = dataY[label_idx[a:b]]
            # print ('label_idx[a:b]', label_idx[a:b])
            label_arr += [labels] * (j - i)
            image_idx_arr += list(label_idx[a:b])
        batch_num_arr += [batch_num + 1] * len(batchX)
        dataBatchX[batch_num, :] = batchX
        dataBatchY[batch_num, :] = batchY
    
    if statsFileName:
        batch_num_image_idx_info = np.column_stack((
            np.array(image_idx_arr).reshape(-1, 1),
            np.array(batch_num_arr).reshape(-1, 1),
            np.array(label_arr).reshape(-1, 1)
        ))
        batch_num_image_idx_info = pd.DataFrame(batch_num_image_idx_info,
                                                columns=['index',
                                                         'batch_num',
                                                         'image_label'])
        if not os.path.exists(path_dict['batchFolderPath']):
            os.makedirs(path_dict['batchFolderPath'])
        path = os.path.join(path_dict['batchFolderPath'], statsFileName)
        batch_num_image_idx_info.to_csv(path, index=None)
    
    if fileName:
        logging.info('The Data batches dumped has shape: %s', str(dataBatchX.shape))
        logging.info('The Label batch dumped has shape: %s', str(dataBatchY.shape))
        dumpPickleFile(dataX=dataBatchX,
                              dataY=dataBatchY,
                              labelDict=None,
                              folderPath=path_dict['batchFolderPath'],
                              picklefileName=fileName)


