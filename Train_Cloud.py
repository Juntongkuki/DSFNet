import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import os.path as osp
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import argparse
from datetime import datetime
from utils.loss import BCEDiceLoss
from utils.dataloader import get_loader
from utils.utils import AvgMeter
import torch.nn.functional as F
from lib.DSFNet import DSFNet
import random
import numpy as np
from tqdm import tqdm
import timm
torch.autograd.set_detect_anomaly(True)



def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 42
seed_everything(seed)


def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()

        # ---- data prepare ----
        images, gts = pack['image'], pack['label']

        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- prediction ----
        pred = model(x=images, flag='train')

        # ---- loss function ----
        loss5 = BCEDiceLoss(pred[5], gts)
        loss4 = BCEDiceLoss(pred[4], gts)
        loss3 = BCEDiceLoss(pred[3], gts)
        loss2 = BCEDiceLoss(pred[2], gts)
        loss1 = BCEDiceLoss(pred[1], gts)
        loss0 = BCEDiceLoss(pred[0], gts)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 +loss5

        # ---- backward ----
        with torch.autograd.detect_anomaly():
            loss.backward()

        optimizer.step()


        # ---- recording loss ----
        loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[loss: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))

    save_path = opt.save_root
    os.makedirs(save_path, exist_ok=True)

    if epoch % 50 == 0:
        torch.save(model.state_dict(), osp.join(opt.save_root, str(opt.model) + '_' + str(opt.dataset) + '_' + str(epoch) + '.pth'))
        print('[Saving Snapshot:]', save_path + 'model-%d.pth' % epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=300, help='epoch number')
    parser.add_argument('--model', type=str,
                        default="baseline", help='the name of the model used')
    parser.add_argument('--dataset', type=str,
                        default="EndoScene", help='the name of dataset used')  # the name of the dataset
    parser.add_argument('--lr', type=float,
                        default=1e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=224, help='training dataset size')
    parser.add_argument('--train_path', type=str,       #the path of dataset
                        default='/root/DSFNet/imgs/data', help='path to train dataset')

    parser.add_argument('--save_root', type=str,             #the path of saving model's trained parameters
                        default='/root/DSFNet/snapshots/')
    parser.add_argument('--gpu', type=str,
                        default='1', help='used GPUs')
    opt = parser.parse_args()

    # ---- build models ----
    image_root = '{}/EndoScene/TrainDataset/images/'.format(opt.train_path)  # The folder where the training images are placed
    gt_root = '{}/EndoScene/TrainDataset/masks/'.format(opt.train_path) # The folder where the labels are placed
    torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    model = DSFNet()
    model = nn.DataParallel(model).cuda()

    params = model.parameters()

    optimizer = torch.optim.Adam(params, opt.lr)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    
    for epoch in tqdm(range(1, opt.epoch+1)):
        train(train_loader, model, optimizer, epoch)



