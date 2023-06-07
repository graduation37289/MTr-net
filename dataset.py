import os
import random
import cv2
import numpy as np
import torch
import torch.utils.data


class DatasetNIH(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, input_w, input_h, input_m, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.input_w = input_w
        self.input_h = input_h
        self.input_m = input_m

    def __len__(self):
        return len(self.img_ids)
    
    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))
        
        tumor = (label==2)
        background = np.logical_not(tumor)
        
        results_map[0, :, :, :] = np.where(tumor, 1, 0)
        results_map[1, :, :, :] = np.where(background, 1, 0)
        return results_map
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))
        
        img = np.array([img])
        mask = np.array([mask])
#         b, img_h, img_w, img_d = mask.shape
        
#         h_off = random.randint(0, img_h - self.input_w)
#         w_off = random.randint(0, img_w - self.input_h)
        
#         img = img[:, h_off: h_off + self.input_w, w_off: w_off + self.input_h, : ]
#         mask = mask[:, h_off: h_off + self.input_w, w_off: w_off + self.input_h, : ]
        
#         randi = np.random.rand(1)
#         if randi <= 0.3:
#             pass
#         elif randi <= 0.5:
#             img = img[:, :, ::-1, :]
#             mask = mask[:, :, ::-1, :]
#         elif randi <= 0.8:
#             img = img[:, ::-1, :, :]
#             mask = mask[:, ::-1, :, :]
#         else:
#             img = img[:, ::-1, ::-1, :]
#             mask = mask[:, ::-1, ::-1, :]
        
        img = img.astype('float32')
        mask = mask.astype('float32')
        
        return img, mask, {'img_id': img_id}


class DatasetMSD(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, input_w, input_h, input_m, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.input_w = input_w
        self.input_h = input_h
        self.input_m = input_m

    def __len__(self):
        return len(self.img_ids)
    
    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))
        
        background = (label==0)
        pancreas = np.logical_not(background)
        
        results_map[0, :, :, :] = np.where(pancreas, 1, 0)
        results_map[1, :, :, :] = np.where(background, 1, 0)
        return results_map
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
#         print('self.img_dir',self.img_dir)
#         print('img_id',img_id)
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))
        mask = self.id2trainId(mask)
        
        img = np.array([img])
        mask = np.array([mask])
        
#         mask[mask == 2] = 1
        
        img = img.astype('float32')
        mask = mask.astype('float32')
        mask = np.squeeze(mask)
        
        return img, mask, {'img_id': img_id}
    
class DatasetMSD2d(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, input_w, input_h, input_m, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.input_w = input_w
        self.input_h = input_h
        self.input_m = input_m

    def __len__(self):
        return len(self.img_ids)
    
    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1]))
        
        background = (label==0)
        pancreas = np.logical_not(background)
        
        results_map[0, :, :] = np.where(pancreas, 1, 0)
        results_map[1, :, :] = np.where(background, 1, 0)
        return results_map
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
#         print('self.img_dir',self.img_dir)
#         print('img_id',img_id)
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))
        mask = self.id2trainId(mask)
        
        img = np.array([img])
        mask = np.array([mask])
        
#         mask[mask == 2] = 1
        
        img = img.astype('float32')
        mask = mask.astype('float32')
        mask = np.squeeze(mask)
        
        return img, mask, {'img_id': img_id}
    
class Datasettumour2d(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, input_w, input_h, input_m, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.input_w = input_w
        self.input_h = input_h
        self.input_m = input_m
    
    def __len__(self):
        return len(self.img_ids)
    
    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1]))
        
        tumor = (label==2)
        background = np.logical_not(tumor)
        
        results_map[0, :, :] = np.where(tumor, 1, 0)
        results_map[1, :, :] = np.where(background, 1, 0)
        return results_map
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
#         print('self.img_dir',self.img_dir)
#         print('img_id',img_id)
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))
        mask = self.id2trainId(mask)
        
        img = np.array([img])
        mask = np.array([mask])
        
#         mask[mask == 2] = 1
        
        img = img.astype('float32')
        mask = mask.astype('float32')
        mask = np.squeeze(mask)
        
        return img, mask, {'img_id': img_id}



class Datasettumour(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, input_w, input_h, input_m, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.input_w = input_w
        self.input_h = input_h
        self.input_m = input_m

    def __len__(self):
        return len(self.img_ids)
    
    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))
        
        tumor = (label==2)
        background = np.logical_not(tumor)
        
        results_map[0, :, :, :] = np.where(tumor, 1, 0)
        results_map[1, :, :, :] = np.where(background, 1, 0)
        return results_map
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
#         print('self.img_dir',self.img_dir)
#         print('img_id',img_id)
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))
#         print('mask ', mask.shape)
        mask = self.id2trainId(mask)
#         print('mask ', mask.shape)
        
        img = np.array([img])
        mask = np.array([mask])
#         print('img, mask array', img.shape, mask.shape)
        
        img = img.astype('float32')
        mask = mask.astype('float32')
        mask = np.squeeze(mask)
#         print('img, mask astype', img.shape, mask.shape)
        
        return img, mask, {'img_id': img_id}


    
class Datasetspleent(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, input_w, input_h, input_m, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.input_w = input_w
        self.input_h = input_h
        self.input_m = input_m

    def __len__(self):
        return len(self.img_ids)
    
    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))
        
        tumor = (label==2)
        background = np.logical_not(tumor)
        
        results_map[0, :, :, :] = np.where(tumor, 1, 0)
        results_map[1, :, :, :] = np.where(background, 1, 0)
        return results_map
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))
        
        img = np.array([img])
        mask = np.array([mask])
#         b, img_h, img_w, img_d = mask.shape
        
#         h_off = random.randint(0, img_h - self.input_w)
#         w_off = random.randint(0, img_w - self.input_h)
        
#         img = img[:, h_off: h_off + self.input_w, w_off: w_off + self.input_h, : ]
#         mask = mask[:, h_off: h_off + self.input_w, w_off: w_off + self.input_h, : ]
        
        randi = np.random.rand(1)
        if randi <= 0.3:
            pass
        elif randi <= 0.5:
            img = img[:, :, ::-1, :]
            mask = mask[:, :, ::-1, :]
        elif randi <= 0.8:
            img = img[:, ::-1, :, :]
            mask = mask[:, ::-1, :, :]
        else:
            img = img[:, ::-1, ::-1, :]
            mask = mask[:, ::-1, ::-1, :]
        
        img = img.astype('float32')
        mask = mask.astype('float32')
        
        return img, mask, {'img_id': img_id}


class Datasetspleenv(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, input_w, input_h, input_m, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.input_w = input_w
        self.input_h = input_h
        self.input_m = input_m

    def __len__(self):
        return len(self.img_ids)
    
    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))
        
        tumor = (label==2)
        background = np.logical_not(tumor)
        
        results_map[0, :, :, :] = np.where(tumor, 1, 0)
        results_map[1, :, :, :] = np.where(background, 1, 0)
        return results_map
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))
        
        img = np.array([img])
        mask = np.array([mask])

        img = img.astype('float32')
        mask = mask.astype('float32')
        
        return img, mask, {'img_id': img_id}