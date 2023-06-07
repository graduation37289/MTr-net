import nibabel as nib
import os
import csv
import math
import numpy as np
from medpy import metric
import matplotlib.pyplot as plt
from metrics import iou_score
from utils import AverageMeter
from skimage.measure import label as method_label
from skimage.measure import regionprops
from scipy import ndimage
import GeodisTK

# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge

def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


# 平均表面距离
def binary_assd(s, g, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd

def compute_class_sens_spec(pred, label):
    """
    Compute sensitivity and specificity for a particular example
    for a given class for binary.
    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (height, width, depth).
        label (np.array): binary array of labels, shape is
                          (height, width, depth).
    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """
    tp = np.sum((pred == 1) & (label == 1))
    tn = np.sum((pred == 0) & (label == 0))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))

    SEN = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return SEN, specificity, precision, recall

def liantong(predict, threshold_point):
    img_label, num = method_label(predict, neighbors=8, return_num=True)#输出二值图像中所有的连通域
    props = regionprops(img_label)#输出连通域的属性，包括面积等
    resMatrix = np.zeros(img_label.shape)
    if len(props)>1:
        for i in range(0, len(props)):
            if props[i].area > threshold_point:
                tmp = (img_label == i+1).astype(np.uint8)
                resMatrix += tmp #组合所有符合条件的连通域
        return resMatrix
    else:
        return predict

def pad_3d(image, padval, xmax, ymax, XMAX, YMAX):
        npad = ((int(xmax[0]), XMAX-int(xmax[1])), (int(ymax[0]), YMAX-int(ymax[1])), (0,0))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)
        return padded
    
def pad_2d(image, padval, xmax, ymax, XMAX, YMAX):
        npad = ((int(xmax[0]), XMAX-int(xmax[1])), (int(ymax[0]), YMAX-int(ymax[1])))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)
        return padded

dataname = 'pancreas_UNext_woDS25d/'
fold = 'fold1'

all_path = './outputs/' + dataname + fold 
save_path25 = './outputs/' + dataname + fold + '/25d/all/'
save_pathwei3d = './outputs/' + dataname + fold + '/wei3d/all/'
save_path2d = './outputs/' + dataname + fold + '/2d/all/'
save_path = './outputs/' + dataname + fold + '/post/'

predict_path = all_path + '/25d/0/predict/'
label_path = all_path + 'label/'

bbox_path = 'data/pancreas'

list_win = os.path.join(all_path, 'win' + '.txt')
output = open(list_win, 'w')

classes = 2
areas = 180
if not os.path.exists(save_path25):
        os.makedirs(save_path25)
if not os.path.exists(save_pathwei3d):
        os.makedirs(save_pathwei3d)
if not os.path.exists(save_path2d):
        os.makedirs(save_path2d)
if not os.path.exists(save_path):
        os.makedirs(save_path)

# diceslice_path = save_path + 'diceslice' + '.txt'  # 也可以创建一个.doc的word文档
# diceslice = open(diceslice_path, 'w')
# dicemerge_path = save_path + 'dicemerge' + '.txt'  # 也可以创建一个.doc的word文档
# dicemerge = open(dicemerge_path, 'w')
iou_avg_meter = AverageMeter()
dice_avg_meter = AverageMeter()
pre2d_avg_meter = AverageMeter()
recall2d_avg_meter = AverageMeter()

list_25d = os.path.join(all_path, '25d' + '.txt')
output25d = open(list_25d, 'w')
output25d.write('name/iou/dice/precision/recall/HD95/ASSD/SEN/specificity' + '\n')

list_result = os.path.join(all_path, 'zongqie' + '.txt')
outputqie = open(list_result, 'w')
outputqie.write('name/iou/dice/precision/recall/HD95/ASSD/SEN/specificity' + '\n')


IDpredict = os.listdir(predict_path)
for idpredict in sorted(IDpredict):
    print('idpredict', idpredict)
    bbox_t25 = [i_id for i_id in open(bbox_path + '/25d/' + fold + '/list/00' + str(idpredict) + '.txt')]
    bbox_twei3d = [i_id for i_id in open(bbox_path + '/wei3d/' + fold + '/list/00' + str(idpredict) + '.txt')]
    bbox_t2d = [i_id for i_id in open(bbox_path + '/2d/' + fold + '/list/00' + str(idpredict) + '.txt')]
    
    predict_copy25 = np.zeros((512,512,len(bbox_t25)+3))
    label_copy25 = np.zeros((512,512,len(bbox_t25)+3))
    predict_copywei3d = np.zeros((512,512,len(bbox_twei3d)*4))
    label_copywei3d = np.zeros((512,512,len(bbox_twei3d)*4))
    predict_copy2d = np.zeros((512,512,len(bbox_t2d)))
    label_copy2d = np.zeros((512,512,len(bbox_t2d)))
    
    predict_copy = np.zeros((512,512,predict_copy25.shape[2]))
    
    sum = 0
    for bbox25 in bbox_t25:
        index = bbox25.split("/")
        
        prename25 = all_path + '/25d/0/predict/' + str(idpredict) + '/' + 'PANCREAS_00' + str(idpredict) + '_slice' + '{:0>4}'.format(index[0]) + 'predict.nii.gz'
        labname25 = all_path + '/25d/0/label/' + str(idpredict) + '/' + 'PANCREAS_00' + str(idpredict) + '_slice' + '{:0>4}'.format(index[0]) + 'label.nii.gz'
        
        predict = nib.load(prename25).get_data()
        predict = np.array(predict)
#         print('predict', predict.shape)
#         print('labelpath', Label_path + labelname)
        label = nib.load(labname25).get_data()
        label = np.array(label)
        
        predict = pad_3d(predict, 0, [index[1], index[2]], [index[3], index[4]], 512, 512)
        label = pad_3d(label, 0, [index[1], index[2]], [index[3], index[4]], 512, 512)
        
        predict_copy25[:, :, sum:sum+4] = predict_copy25[:, :, sum:sum+4] + predict
        label_copy25[:, :, sum:sum+4] = label_copy25[:, :, sum:sum+4] + label
        
        sum = sum + 1
    
    slice_number = predict_copy25.shape[2]
    for j in range(0, slice_number):
            if j == 1 or j == slice_number-2:
                predict_ = predict_copy25[:, :, j]
                label_ = label_copy25[:, :, j]
                predict_[predict_ == 1] = 0
                predict_[predict_ == 2] = 1
                label_[label_ == 2] = 1
            if j == 2 or j == slice_number-3:
                predict_ = predict_copy25[:, :, j]
                label_ = label_copy25[:, :, j]
                predict_[predict_ == 1] = 0
                predict_[predict_ == 2] = 1
                predict_[predict_ == 3] = 1
                label_[label_ == 2] = 1
                label_[label_ == 3] = 1
            if j > 2 and j < slice_number-3:
                predict_ = predict_copy25[:, :, j]
                label_ = label_copy25[:, :, j]
                predict_[predict_ == 1] = 0
                predict_[predict_ == 2] = 1
                predict_[predict_ == 3] = 1
                predict_[predict_ == 4] = 1
                label_[label_ == 2] = 1
                label_[label_ == 3] = 1
                label_[label_ == 4] = 1
            predict_copy25[:, :, j] = liantong(predict_copy25[:, :, j], areas)
    
    new_predict25 = nib.Nifti1Image(predict_copy25, np.eye(4, 4))
    nib.save(new_predict25, save_path25 + str(idpredict) +'pred.nii.gz')
    new_label25 = nib.Nifti1Image(label_copy25, np.eye(4, 4))
    nib.save(new_label25, save_path25 + str(idpredict) +'label.nii.gz')
    
    sum = 0
    for bboxwei3d in bbox_twei3d:
        index = bboxwei3d.split("/")
        
        prenamewei3d = all_path + '/wei3d/0/predict/' + str(idpredict) + '/' + 'PANCREAS_00' + str(idpredict) + '_slice' + '{:0>4}'.format(index[0]) + 'predict.nii.gz'
        labnamewei3d = all_path + '/wei3d/0/label/' + str(idpredict) + '/' + 'PANCREAS_00' + str(idpredict) + '_slice' + '{:0>4}'.format(index[0]) + 'label.nii.gz'
        
        predict = nib.load(prenamewei3d).get_data()
        predict = np.array(predict)
#         print('predict', predict.shape)
#         print('labelpath', Label_path + labelname)
        label = nib.load(labnamewei3d).get_data()
        label = np.array(label)
        
        predict = pad_3d(predict, 0, [index[1], index[2]], [index[3], index[4]], 512, 512)
        label = pad_3d(label, 0, [index[1], index[2]], [index[3], index[4]], 512, 512)
        
        predict_copywei3d[:, :, sum:sum+4] = predict_copywei3d[:, :, sum:sum+4] + predict
        label_copywei3d[:, :, sum:sum+4] = label_copywei3d[:, :, sum:sum+4] + label
        
        sum = sum + 4
    
    new_predictwei3d = nib.Nifti1Image(predict_copywei3d, np.eye(4, 4))
    nib.save(new_predictwei3d, save_pathwei3d + str(idpredict) +'pred.nii.gz')
    new_labelwei3d = nib.Nifti1Image(label_copywei3d, np.eye(4, 4))
    nib.save(new_labelwei3d, save_pathwei3d + str(idpredict) +'label.nii.gz')
    
    
    sum = 0
    for bbox2d in bbox_t2d:
        index = bbox2d.split("/")
        
        prename2d = all_path + '/2d/0/predict/' + str(idpredict) + '/' + 'PANCREAS_00' + str(idpredict) + '_slice' + '{:0>4}'.format(index[0]) + 'predict.nii.gz'
        labname2d = all_path + '/2d/0/label/' + str(idpredict) + '/' + 'PANCREAS_00' + str(idpredict) + '_slice' + '{:0>4}'.format(index[0]) + 'label.nii.gz'
        
        predict = nib.load(prename2d).get_data()
        predict = np.array(predict)
#         print('predict', predict.shape)
#         print('labelpath', Label_path + labelname)
        label = nib.load(labname2d).get_data()
        label = np.array(label)
        
        predict = pad_2d(predict, 0, [index[1], index[2]], [index[3], index[4]], 512, 512)
        label = pad_2d(label, 0, [index[1], index[2]], [index[3], index[4]], 512, 512)
        
        predict_copy2d[:, :, sum] = predict_copy2d[:, :, sum] + predict
        label_copy2d[:, :, sum] = label_copy2d[:, :, sum] + label
        
        sum = sum + 1
    
    new_predict2d = nib.Nifti1Image(predict_copy2d, np.eye(4, 4))
    nib.save(new_predict2d, save_path2d + str(idpredict) +'pred.nii.gz')
    new_label2d = nib.Nifti1Image(label_copy2d, np.eye(4, 4))
    nib.save(new_label2d, save_path2d + str(idpredict) +'label.nii.gz')
    
    SEN2d, specificity2d, precision2d, recall2d = compute_class_sens_spec(predict_copy2d, label_copy2d)
    pre2d_avg_meter.update(precision2d)
    recall2d_avg_meter.update(recall2d)
    
    #方法二取wei3d和2d方法交集，弥补到25d方法
    for i in range(0, predict_copy25.shape[2]):
        predict25 = predict_copy25[:, :, i]
        if i < predict_copywei3d.shape[2]:
            predictwei3d = predict_copywei3d[:, :, i]
        predict2d = predict_copy2d[:, :, i]
        if i < predict_copywei3d.shape[2]:
            predict_ = predictwei3d + predict2d
            predict_[predict_ == 1] = 0
            
            predict = predict25 + predict_
            predict[predict == 2] = 1
            predict[predict == 3] = 1
        else:
            predict = predict25
        predict_copy[:, :, i] = predict_copy[:, :, i] + predict
    
    iou25,dice25 = iou_score(predict_copy25, label_copy25)
    print('iou25, dice25', iou25, dice25)
    SEN25, specificity25, precision25, recall25 = compute_class_sens_spec(predict_copy25, label_copy25)
    HD9525 = binary_hausdorff95(predict_copy25.astype('float32'), label_copy25.astype('float32'))
    ASD25 = binary_assd(predict_copy25.astype('float32'), label_copy25.astype('float32'))
    output25d.write(str(idpredict) + '/' + str(iou25) + '/' + str(dice25) + '/' + str(precision25) + '/' + str(recall25) + '/' + str(HD9525) + '/' + str(ASD25) + '/' + str(SEN25) + '/' + str(specificity25) + '\n')
    
    iou,dice = iou_score(predict_copy, label_copy25)
    print('iou, dice', iou, dice)
    SEN, specificity, precision, recall = compute_class_sens_spec(predict_copy, label_copy25)
    HD95 = binary_hausdorff95(predict_copy.astype('float32'), label_copy25.astype('float32'))
    ASD = binary_assd(predict_copy.astype('float32'), label_copy25.astype('float32'))
    
    outputqie.write(str(idpredict) + '/' + str(iou) + '/' + str(dice) + '/' + str(precision) + '/' + str(recall) + '/' + str(HD95) + '/' + str(ASD) + '/' + str(SEN) + '/' + str(specificity) + '\n')
    
    iou_avg_meter.update(iou)
    dice_avg_meter.update(dice)
    
    new_predict = nib.Nifti1Image(predict_copy, np.eye(4, 4))
    nib.save(new_predict, save_path + str(idpredict) +'pred.nii.gz')
    new_label = nib.Nifti1Image(label_copy25, np.eye(4, 4))
    nib.save(new_label, save_path + str(idpredict) +'label.nii.gz')
    
print('IoU: %.4f' % iou_avg_meter.avg)
print('Dice: %.4f' % dice_avg_meter.avg)
print('pre2d: %.4f' % pre2d_avg_meter.avg)
print('recall2d: %.4f' % recall2d_avg_meter.avg)
output.write(str(iou_avg_meter.avg) + '   ' + str(dice_avg_meter.avg) + '\n')
output.close()
outputqie.close()
output25d.close()

