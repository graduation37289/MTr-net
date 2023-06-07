import argparse
import os
from glob import glob
import nibabel as nib
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import TrNet2D
from dataset import DatasetNIH
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
import numpy as np
from TrNet2D import TrNet2D


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    fold = 'fold1'
    with open('models/%s/%s/%s/config.yml' % (args.name, fold, '2d'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = TrNet2D.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()
    val_img_ids = []
    train_img_ids = []
    
    # Data loading code
    img_idsval = glob(os.path.join('data', config['dataset'], '2d', fold, 'images', 'test', '*' + config['img_ext']))
    img_idsval = [os.path.splitext(os.path.basename(p))[0] for p in img_idsval]
    
    for idval in img_idsval:
            val_img_ids.append(idval)
    
    
    model.load_state_dict(torch.load('models/%s/%s/%s/model.pth' %
                                     (args.name, fold, '2d')))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = DatasetNIH(
        img_ids=val_img_ids,
        img_dir=os.path.join('data', config['dataset'], '2d', fold, 'images', 'test'),
        mask_dir=os.path.join('data', config['dataset'], '2d', fold, 'masks', 'test'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_w = config['input_w'],
        input_h = config['input_h'],
        input_m = config['input_m'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)
            
            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            
            output = torch.sigmoid(output).cpu().numpy()
            target = target.cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0
            
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    num = meta['img_id'][i][-12:-10]
                    path1 = './outputs/' + config['name'] + '/' + fold + '/2d/' + str(c) + '/predict/' + str(num) + '/'
                    path2 = './outputs/' + config['name'] + '/' + fold + '/2d/' + str(c) + '/label/' + str(num) + '/'
                    if not os.path.exists(path1):
                        os.makedirs(path1)
                    if not os.path.exists(path2):
                        os.makedirs(path2)
                    
                    new_image = nib.Nifti1Image(output[i, c], np.eye(4, 4))
                    nib.save(new_image, os.path.join('outputs', config['name'], fold, '2d', str(c), 'predict', str(meta['img_id'][i][-12:-10]),   meta['img_id'][i] + 'predict.nii.gz'))
                    
                    new_label = nib.Nifti1Image(target[i, c], np.eye(4, 4))
                    nib.save(new_label, os.path.join('outputs', config['name'], fold, '2d', str(c), 'label', str(meta['img_id'][i][-12:-10]),   meta['img_id'][i] + 'label.nii.gz'))
        
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
