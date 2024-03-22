import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os, argparse
import os.path as osp
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy import misc
from utils.dataloader import test_dataset
from utils.eval import Evaluator
from tqdm import tqdm
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from lib.DSFNet import DSFNet
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.")

# pred
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/DSFNet_before/{}/13_04_300.pth') #change to your own parameters document
parser.add_argument('--data_root', type=str, default='/root/DSFNet/imgs/data/{}/TestDataset')
parser.add_argument('--save_root', type=str, default='/root/DSFNet/snapshots/memory_pre/{}/')
parser.add_argument('--gpu', type=str, default='0')

for _data_name in ['EndoScene']:
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    data_path = opt.data_root.format(_data_name)
    save_path = opt.save_root.format(_data_name)
    load_path = opt.pth_path.format(_data_name)
    
    model = DSFNet()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(load_path),False)
    model = model.cuda()
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    evaluator = Evaluator()
    
    bar = tqdm(test_loader.images)  
    for i in bar:
        image, gt, name = test_loader.load_data()
        image = image.cuda()
        gt = gt.cuda()

        pred = model(image, 'test')
        res = pred[4].sigmoid()
        res = F.upsample(res, size=(gt.shape[2], gt.shape[3]), mode='bilinear', align_corners=False)

        evaluator.update(res, gt)
        
        res = res.sigmoid().data.cpu().numpy().squeeze()
        imageio.imsave(save_path+name, res)
    evaluator.show()









