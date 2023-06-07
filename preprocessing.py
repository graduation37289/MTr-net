import os
import time
import math
from glob import glob
import nibabel as nib

from numpy import random

import numpy as np
import csv


def pre_precessing(image):
        image[image <= -100] = -100
        image[image >= 240] = 240
        image += 100
        image = image / 340
        return image


name = ''  # savepath
fold = ''
image_path = ''  # npy_path
label_path = ''
margin = 0

YMAX = 224    
XMAX = 224
ZMAX = 1
plane = 'Z'

# savelist_path = 'pancreas/'+ fold + '/lists'
# if not os.path.exists(savelist_path):  
#     os.makedirs(savelist_path) 

low_range = -100
high_range = 240


########fold1########
train_list = [i_id.strip().split() for i_id in open(name + 'image_'+ fold + '_train.txt')]
test_list = [i_id.strip().split() for i_id in open(name + 'image_'+ fold + '_test.txt')]

Sum = 0
for train_path in train_list:
    Sum = Sum + 1
    train_path = train_path[0]
    image = np.load(train_path)
    label = np.load(label_path + 'label' + train_path[-8:])
    
    width = label.shape[0]
    height = label.shape[1]
    
    slice_number = label.shape[2]
    z = min(np.nonzero(label)[2])
    for j in range(0, slice_number):
        
        
        train_file = os.path.join(name, '2d', fold, 'images', 'train')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        train_file = os.path.join(name, '25d', fold, 'images', 'train')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        train_file = os.path.join(name, 'wei3d', fold, 'images', 'train')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        
        val_file = os.path.join(name, '2d', fold, 'images', 'val')
        if not os.path.exists(val_file):  
                os.makedirs(val_file)
        val_file = os.path.join(name, '25d', fold, 'images', 'val')
        if not os.path.exists(val_file):  
                os.makedirs(val_file)
        val_file = os.path.join(name, 'wei3d', fold, 'images', 'val')
        if not os.path.exists(val_file):  
                os.makedirs(val_file)
                
        train_file = os.path.join(name, '2d', fold, 'masks', 'train')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        train_file = os.path.join(name, '25d', fold, 'masks', 'train')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        train_file = os.path.join(name, 'wei3d', fold, 'masks', 'train')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
                
        val_file = os.path.join(name, '2d', fold, 'masks', 'val')
        if not os.path.exists(val_file):  
                os.makedirs(val_file)
        val_file = os.path.join(name, '25d', fold, 'masks', 'val')
        if not os.path.exists(val_file):  
                os.makedirs(val_file)
        val_file = os.path.join(name, 'wei3d', fold, 'masks', 'val')
        if not os.path.exists(val_file):  
                os.makedirs(val_file)
        
        image_filename = train_path.split('/')[2][:-4] + '_' + 'slice' + '{:0>4}'.format(j) + '.npy'
        
        label_filename = train_path.split('/')[2][:-4] + '_' + 'slice' + '{:0>4}'.format(j) + '.npy'
        
        
        image2d = image[:, :, j]
        label2d = label[:, :, j]
        image2d = pre_precessing(image2d)
        arr2d = np.nonzero(label2d)
        
        if j < slice_number-3:
            image25 = image[:, :, j:j+4]
            label25 = label[:, :, j:j+4]
            label1 = label[:, :, j]
            label3 = label[:, :, j+3]
            image25 = pre_precessing(image25)
            arr251 = np.nonzero(label1)
            arr253 = np.nonzero(label3)
        
        
        if len(arr2d[0]) > 0:
                minA = np.min(arr2d[0])
                maxA = np.max(arr2d[0])
                    
                minB = np.min(arr2d[1])
                maxB = np.max(arr2d[1])
                    
                        
                if maxA - minA < XMAX:
                        bcA = int((XMAX - (maxA - minA))/2)
                        maxA = maxA + bcA
                        minA = minA - bcA

                if maxB - minB < YMAX:
                        bcB = int((YMAX - (maxB - minB))/2)
                        maxB = maxB + bcB
                        minB = minB - bcB

                if maxA - minA < XMAX:
                        chax = XMAX - (maxA - minA)
                        maxA = maxA + chax
                if maxB - minB < YMAX:
                        chay = YMAX - (maxB - minB)
                        maxB = maxB + chay
                if maxA - minA > XMAX:
                        chax = maxA - minA - XMAX
                        maxA = maxA + chax
                if maxB - minB > YMAX:
                        chay = maxB - minB - YMAX
                        maxB = maxB + chay
                
                if minA < 0:
                        maxA = maxA + abs(minA)
                        minA = 0
                if minB < 0:
                        maxB = maxB + abs(minB)
                        minB = 0
                if maxA > width:
                        minA = minA - (maxA - width)
                        maxA = width               
                if maxB > height:
                        minB = minB - (maxB - height)
                        maxB = height
                
                image2d = image2d[max(minA, 0): min(maxA, width), max(minB, 0): min(maxB, height)]
                label2d = label2d[max(minA, 0): min(maxA, width), max(minB, 0): min(maxB, height)]
                
                np.save(name + '/2d/'+ fold +'/images/train/' + image_filename, image2d)
                np.save(name + '/2d/'+ fold +'/masks/train/' + label_filename, label2d)
                if Sum % 5 == 0:
                    np.save(name + '/2d/'+ fold +'/images/val/' + image_filename, image2d)
                    np.save(name + '/2d/'+ fold +'/masks/val/' + label_filename, label2d)
        
        if len(arr251[0]) > 0 and len(arr253[0]) > 0:
                axes_index = np.argwhere(label25 == 1)
                one, two, three = axes_index[:, 0], axes_index[:, 1], axes_index[:, 2]
                    
                minA = np.min(one)
                maxA = np.max(one)
                    
                minB = np.min(two)
                maxB = np.max(two)
                        
                if maxA - minA < XMAX:
                        bcA = int((XMAX - (maxA - minA))/2)
                        maxA = maxA + bcA
                        minA = minA - bcA

                if maxB - minB < YMAX:
                        bcB = int((YMAX - (maxB - minB))/2)
                        maxB = maxB + bcB
                        minB = minB - bcB

                if maxA - minA < XMAX:
                        chax = XMAX - (maxA - minA)
                        maxA = maxA + chax
                if maxB - minB < YMAX:
                        chay = YMAX - (maxB - minB)
                        maxB = maxB + chay
                if maxA - minA > XMAX:
                        chax = maxA - minA - XMAX
                        maxA = maxA + chax
                if maxB - minB > YMAX:
                        chay = maxB - minB - YMAX
                        maxB = maxB + chay
                
                if minA < 0:
                        maxA = maxA + abs(minA)
                        minA = 0
                if minB < 0:
                        maxB = maxB + abs(minB)
                        minB = 0
                if maxA > width:
                        minA = minA - (maxA - width)
                        maxA = width               
                if maxB > height:
                        minB = minB - (maxB - height)
                        maxB = height
                        
                image25 = image25[max(minA, 0): min(maxA, width), max(minB, 0): min(maxB, height), :]
                label25 = label25[max(minA, 0): min(maxA, width), max(minB, 0): min(maxB, height), :]
                
                if z == j:
                    np.save(name + '/wei3d/'+ fold +'/images/train/' + image_filename, image25)
                    np.save(name + '/wei3d/'+ fold +'/masks/train/' + label_filename, label25)
                    z = z + 4
                
                np.save(name + '/25d/'+ fold +'/images/train/' + image_filename, image25)
                np.save(name + '/25d/'+ fold +'/masks/train/' + label_filename, label25)
                if Sum % 5 == 0:
                    np.save(name + '/25d/'+ fold +'/images/val/' + image_filename, image25)
                    np.save(name + '/25d/'+ fold +'/masks/val/' + label_filename, label25)        
                
for test_path in test_list:
    test_path = test_path[0]
    print('test_path ', test_path)
    image = np.load(test_path)
    label = np.load(label_path + 'label' + test_path[-8:])
    
    z = min(np.nonzero(label)[2])
    
    list_file = os.path.join(name, '25d', fold, 'list')
    if not os.path.exists(list_file):  
            os.makedirs(list_file)
    list_filewei3d = os.path.join(name, 'wei3d', fold, 'list')
    if not os.path.exists(list_filewei3d):  
            os.makedirs(list_filewei3d)
    list_file2d = os.path.join(name, '2d', fold, 'list')
    if not os.path.exists(list_file2d):  
            os.makedirs(list_file2d)        
    
    
    list_id = os.path.join(list_file, test_path[-8:-4] + '.txt')
    output = open(list_id, 'w')
    list_idwei3d = os.path.join(list_filewei3d, test_path[-8:-4] + '.txt')
    outputwei3d = open(list_idwei3d, 'w')
    list_id2d = os.path.join(list_file2d, test_path[-8:-4] + '.txt')
    output2d = open(list_id2d, 'w')
    
    width = label.shape[0]
    height = label.shape[1]
    
    slice_number = label.shape[2]
    for j in range(0, slice_number):
        train_file = os.path.join(name, '2d', fold, 'images', 'test')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        train_file = os.path.join(name, '25d', fold, 'images', 'test')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        train_file = os.path.join(name, 'wei3d', fold, 'images', 'test')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        
                
        train_file = os.path.join(name, '2d', fold, 'masks', 'test')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        train_file = os.path.join(name, '25d', fold, 'masks', 'test')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        train_file = os.path.join(name, 'wei3d', fold, 'masks', 'test')
        if not os.path.exists(train_file):  
                os.makedirs(train_file)
        
        image_filename = test_path.split('/')[2][:-4] + '_' + 'slice' + '{:0>4}'.format(j) + '.npy'
        
        label_filename = test_path.split('/')[2][:-4] + '_' + 'slice' + '{:0>4}'.format(j) + '.npy'
        
        
        image2d = image[:, :, j]
        label2d = label[:, :, j]
        image2d = pre_precessing(image2d)
        arr2d = np.nonzero(label2d)
        
        if j < slice_number-3:
            image25 = image[:, :, j:j+4]
            label25 = label[:, :, j:j+4]
            label1 = label[:, :, j]
            label3 = label[:, :, j+3]
            image25 = pre_precessing(image25)
            arr251 = np.nonzero(label1)
            arr253 = np.nonzero(label3)
        
        
        if len(arr2d[0]) > 0:
                minA = np.min(arr2d[0])
                maxA = np.max(arr2d[0])
                    
                minB = np.min(arr2d[1])
                maxB = np.max(arr2d[1])
                    
                        
                if maxA - minA < XMAX:
                        bcA = int((XMAX - (maxA - minA))/2)
                        maxA = maxA + bcA
                        minA = minA - bcA

                if maxB - minB < YMAX:
                        bcB = int((YMAX - (maxB - minB))/2)
                        maxB = maxB + bcB
                        minB = minB - bcB

                if maxA - minA < XMAX:
                        chax = XMAX - (maxA - minA)
                        maxA = maxA + chax
                if maxB - minB < YMAX:
                        chay = YMAX - (maxB - minB)
                        maxB = maxB + chay
                if maxA - minA > XMAX:
                        chax = maxA - minA - XMAX
                        maxA = maxA + chax
                if maxB - minB > YMAX:
                        chay = maxB - minB - YMAX
                        maxB = maxB + chay
                
                if minA < 0:
                        maxA = maxA + abs(minA)
                        minA = 0
                if minB < 0:
                        maxB = maxB + abs(minB)
                        minB = 0
                if maxA > width:
                        minA = minA - (maxA - width)
                        maxA = width               
                if maxB > height:
                        minB = minB - (maxB - height)
                        maxB = height
                
                image2d = image2d[max(minA, 0): min(maxA, width), max(minB, 0): min(maxB, height)]
                label2d = label2d[max(minA, 0): min(maxA, width), max(minB, 0): min(maxB, height)]
                
                output2d.write(str(j) + '/' + str(minA-npd[0][0]) + '/' + str(maxA+npd[0][1]) + '/' + str(minB-npd[1][0]) + '/' + str(maxB+npd[1][1]) + '\n')
                
                np.save(name + '/2d/'+ fold +'/images/test/' + image_filename, image2d)
                np.save(name + '/2d/'+ fold +'/masks/test/' + label_filename, label2d)
            
        if len(arr251[0]) > 0 and len(arr253[0]) > 0:
                axes_index = np.argwhere(label25 == 1)
                one, two, three = axes_index[:, 0], axes_index[:, 1], axes_index[:, 2]
                    
                minA = np.min(one)
                maxA = np.max(one)
                    
                minB = np.min(two)
                maxB = np.max(two)
                        
                if maxA - minA < XMAX:
                        bcA = int((XMAX - (maxA - minA))/2)
                        maxA = maxA + bcA
                        minA = minA - bcA

                if maxB - minB < YMAX:
                        bcB = int((YMAX - (maxB - minB))/2)
                        maxB = maxB + bcB
                        minB = minB - bcB

                if maxA - minA < XMAX:
                        chax = XMAX - (maxA - minA)
                        maxA = maxA + chax
                if maxB - minB < YMAX:
                        chay = YMAX - (maxB - minB)
                        maxB = maxB + chay
                if maxA - minA > XMAX:
                        chax = maxA - minA - XMAX
                        maxA = maxA + chax
                if maxB - minB > YMAX:
                        chay = maxB - minB - YMAX
                        maxB = maxB + chay

                if minA < 0:
                        maxA = maxA + abs(minA)
                        minA = 0
                if minB < 0:
                        maxB = maxB + abs(minB)
                        minB = 0
                if maxA > width:
                        minA = minA - (maxA - width)
                        maxA = width               
                if maxB > height:
                        minB = minB - (maxB - height)
                        maxB = height
                
                image25 = image25[max(minA, 0): min(maxA, width), max(minB, 0): min(maxB, height), :]
                label25 = label25[max(minA, 0): min(maxA, width), max(minB, 0): min(maxB, height), :]

                output.write(str(j) + '/' + str(minA) + '/' + str(maxA) + '/' + str(minB) + '/' + str(maxB) + '\n')
                
                if z == j:
                    outputwei3d.write(str(j) + '/' + str(minA) + '/' + str(maxA) + '/' + str(minB) + '/' + str(maxB) + '\n')
                    np.save(name + '/wei3d/'+ fold +'/images/test/' + image_filename, image25)
                    np.save(name + '/wei3d/'+ fold +'/masks/test/' + label_filename, label25)
                    z = z + 4
                
                np.save(name + '/25d/'+ fold +'/images/test/' + image_filename, image25)
                np.save(name + '/25d/'+ fold +'/masks/test/' + label_filename, label25)
    output.close()
    outputwei3d.close()
    output2d.close()
        
print('finished!')

