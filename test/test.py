from __future__ import print_function, division
import argparse
from ast import arg
import torch
import torch.nn as nn
import time
import os
import os.path as osp
import yaml
import random
import numpy as np
import scipy.io
import pathlib
import sys
import json
import copy
import multiprocessing as mp
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
# from client import Client
# from server import Server
from utils import set_random_seed
# from data_utils import Data
from util.logger import setup_logger
import math

from utils import get_model, extract_feature
from evaluate import pratice_evaluate

from vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import os
import json
import torch
from random_erasing import RandomErasing
from sampler import RandomIdentitySampler


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='/mnt/dataset1/data',type=str, help='training dir path')
parser.add_argument('--datasets',default='cuhk01,viper,prid,3dpes,ilids,market,MSMT17,cuhk03-np,duke',type=str, help='datasets used')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--kd_train', default=True, type=bool, help='use all training data' )
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--drop_rate', default=0.5, type=float, help='drop rate')

# arguments for federated setting
# parser.add_argument('--local_epoch', default={'market':1, 'MSMT17':5, 'cuhk03-np':1}, type=int, help='number of local epochs')
parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_of_clients', default=9, type=int, help='number of clients')

# arguments for data transformation
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )

# arguments for testing federated model
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--multiple_scale',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--test_dir',default='all',type=str, help='./test_data')
parser.add_argument('--test_per_round', default=10, type=int, help='federated model test per round')


# arguments for vit_pytorch
parser.add_argument('--img_size', default=[256, 128], type=list, help='img_size')
parser.add_argument('--stride_size', default=[16, 16], type=list, help='patch_embed stride size')
parser.add_argument('--drop_path_rate', default=0.1, type=float, help='attn drop path rate')
parser.add_argument('--trans_drop_rate', default=0.0, type=float, help='model drop rate')
parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='model attn rate')
parser.add_argument('--tranformer_type', default='vit_base_patch16_224_TransReID', type=str, help='transformer type')
parser.add_argument('--neck_feat', default='before', type=str, help='feat after BN')

def test():
    args = parser.parse_args()
    log_dir = osp.join(args.project_dir, args.model_name)
    logger = setup_logger('federate_reid_train', log_dir, if_train=True)
    logger.info(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    set_random_seed(1)

    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all, args.kd_train)
    # data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
    data.preprocess()
    
    clients = {}
    client_list = data.client_list
    for cid in client_list:
        clients[cid] = Client(
            cid, 
            data, 
            device,
            args, 
            args.project_dir, 
            args.model_name, 
            args.local_epoch, 
            args.lr, 
            args.batch_size, 
            args.drop_rate, 
            args.stride,
            logger) 

    server = Server(
        clients, 
        data, 
        device,
        args, 
        args.project_dir, 
        args.model_name, 
        args.num_of_clients, 
        args.lr, 
        args.drop_rate, 
        args.stride, 
        args.multiple_scale,
        logger)

    dir_name = os.path.join(args.project_dir, args.model_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


    server.test(use_cuda)


class Client():
    def __init__(self, cid, data, device, args, project_dir, model_name, local_epoch, lr, batch_size, drop_rate, stride, logger):
        self.cid = cid
        self.project_dir = project_dir
        self.model_name = model_name
        self.data = data
        self.device = device
        # self.local_epoch = np.random.randint(low=1, high=10)
        self.local_epoch = local_epoch
        # self.local_epoch = local_epoch[cid]
        self.lr = lr
        self.batch_size = batch_size
        
        self.dataset_sizes = self.data.train_dataset_sizes[cid]
        self.train_loader = self.data.train_loaders[cid]

        self.full_model = get_model(args, self.data.train_class_sizes[cid], drop_rate, stride)
        self.classifier = self.full_model.classifier
        self.full_model.classifier = nn.Sequential()
        # self.full_model.classifier = nn.Linear(768, )
        self.model = self.full_model
        self.distance=0
        self.log_dir = osp.join(args.project_dir, args.model_name)
        self.logger = logger
        self.args = args


class Server():
    def __init__(self, clients, data, device, args, project_dir, model_name, num_of_clients, lr, drop_rate, stride, multiple_scale, logger):
        self.project_dir = project_dir
        self.data = data
        self.device = device
        self.args = args
        self.model_name = model_name
        self.clients = clients
        self.client_list = self.data.client_list
        self.num_of_clients = num_of_clients
        self.lr = lr
        self.multiple_scale = multiple_scale
        self.drop_rate = drop_rate
        self.stride = stride

        self.multiple_scale = []
        for s in multiple_scale.split(','):
            self.multiple_scale.append(math.sqrt(float(s)))

        self.full_model = get_model(args, 751, drop_rate, stride).to(device)
        # self.full_model = get_model(args, 750, drop_rate, stride).to(device)
        self.full_model.classifier = nn.Sequential()
        # self.full_model.classifier.classifier = nn.Sequential()
        self.federated_model=self.full_model
        self.federated_model.eval()
        self.train_loss = []
        self.logger = logger
        self.args = args

    def test(self, use_cuda):
        # print("="*10)
        # print("Start Tesing!")
        # print("="*10)
        # print('We use the scale: %s'%self.multiple_scale)
        # logger = logging.getLogger("="*10)
        self.logger.info("Start Testing!")
        self.logger.info("="*10)
        self.logger.info('We use the scale: %s'%self.multiple_scale)

        for dataset in self.data.datasets:
            self.federated_model = self.federated_model.eval()
            if use_cuda:
                self.federated_model = self.federated_model.to(self.device)
            
            with torch.no_grad():
                gallery_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['gallery'], self.multiple_scale, self.device)
                query_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['query'], self.multiple_scale, self.device)

            result = {
                'gallery_f': gallery_feature.numpy(),
                'gallery_label': self.data.gallery_meta[dataset]['labels'],
                'gallery_cam': self.data.gallery_meta[dataset]['cameras'],
                'query_f': query_feature.numpy(),
                'query_label': self.data.query_meta[dataset]['labels'],
                'query_cam': self.data.query_meta[dataset]['cameras']}

            scipy.io.savemat(os.path.join(self.project_dir,
                        self.model_name,
                        'pytorch_result.mat'),
                        result)
                        
            print(dataset)

            # os.system('python evaluate.py --result_dir {} --dataset {}'.format(os.path.join(self.project_dir, self.model_name), dataset))
            res_dir = os.path.join(self.project_dir, self.model_name)
            pratice_evaluate(res_dir, dataset, self.logger)


def pratice_evaluate(result_dir, dataset, logger):
    result = scipy.io.loadmat(result_dir + '/pytorch_result.mat')
    # result = scipy.io.loadmat(args.result_dir + '/pytorch_result.mat')

    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]


    device = torch.device("cuda:1")
    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    # logger = logging.getLogger("=======evaluate=======")
    # logger.info(dataset+' Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
    # logger.info('-'*15)
    # print()
    # print(args.dataset+' Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
    # print('-'*15)
    # print()


    logger.info(dataset+' Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
    logger.info('-'*15)
    logger.info(' ')


def evaluate(qf, ql, qc, gf, gl, gc, ):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp



def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


def get_model(args, class_sizes, drop_rate, stride):
    model = build_transformer(args, class_sizes, __factory_T_type)
    return model


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, cfg, num_classes, factory):
        super(build_transformer, self).__init__()
        model_path = './.torch/models/jx_vit_base_p16_224-80ecf9dd.pth'
        pretrain_choice = 'imagenet'
        self.neck = 'bnneck'
        self.neck_feat = cfg.neck_feat
        self.in_planes = 768

        self.base = factory[cfg.tranformer_type](img_size=cfg.img_size,
                                                        stride_size=cfg.stride_size, drop_path_rate=cfg.drop_path_rate,
                                                        drop_rate= cfg.trans_drop_rate,
                                                        attn_drop_rate=cfg.attn_drop_rate)
        if cfg.tranformer_type == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):

        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:

            cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss

        if self.neck_feat == 'after':
            # print("Test with feature after BN")
            return feat
        else:
            return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                print('===')
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}



class ImageDataset(Dataset):
    def __init__(self, imgs,  transform = None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data,label = self.imgs[index]
        return self.transform(Image.open(data)), label


class Data():
    def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all, kd_train):
    # def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all):
        self.datasets = datasets.split(',')
        self.batch_size = batch_size
        self.erasing_p = erasing_p
        self.color_jitter = color_jitter
        self.data_dir = data_dir
        self.train_all = '_all' if train_all else ''
        self.kd_train = '_all' if kd_train else ''
        
    def transform(self):
        transform_train = [
                transforms.Resize((256,128), interpolation=3),
                transforms.Pad(10),
                transforms.RandomCrop((256,128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        transform_val = [
                transforms.Resize(size=(256,128),interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        if self.erasing_p > 0:
            transform_train = transform_train + [RandomErasing(probability=self.erasing_p, mode='pixel', max_count=1, device='cpu')]
            # transform_train = transform_train + [RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0])]

        if self.color_jitter:
            transform_train = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train

        self.data_transforms = {
            'train': transforms.Compose(transform_train),
            'val': transforms.Compose(transform_val),
        }

    def preprocess_one_train_dataset(self, dataset):
        """preprocess a training dataset, construct a data loader.
        """
        data_path = os.path.join(self.data_dir, dataset, 'pytorch')
        data_path = os.path.join(data_path, 'train' + self.train_all)
        image_dataset = datasets.ImageFolder(data_path)
        img_per_batch = 64
        num_instance = 4

        loader = torch.utils.data.DataLoader(
            ImageDataset(image_dataset.imgs, self.data_transforms['train']), 
            batch_size=self.batch_size,
            sampler=RandomIdentitySampler(image_dataset.imgs, img_per_batch, num_instance),
            # shuffle=True, 
            # num_workers=0, 
            num_workers=8, 
            pin_memory=True,
            # pin_memory=False,
            collate_fn = train_collate_fn)

        return loader, image_dataset
    

    def preprocess_train(self):
        """preprocess training data, constructing train loaders
        """
        self.train_loaders = {}
        self.train_dataset_sizes = {}
        self.train_class_sizes = {}
        self.client_list = []
        
        # for dataset in self.datasets[:3]:
        for dataset in self.datasets:
            self.client_list.append(dataset)
          
            loader, image_dataset = self.preprocess_one_train_dataset(dataset)

            self.train_dataset_sizes[dataset] = len(image_dataset)
            self.train_class_sizes[dataset] = len(image_dataset.classes)
            self.train_loaders[dataset] = loader
            
        print('Train dataset sizes:', self.train_dataset_sizes)
        print('Train class sizes:', self.train_class_sizes)
        
    def preprocess_test(self):
        """preprocess testing data, constructing test loaders
        """
        self.test_loaders = {}
        self.gallery_meta = {}
        self.query_meta = {}

        for test_dir in self.datasets:
        # test_dir = self.datasets[-1]
            dataset_name = test_dir
            print('test_dir', test_dir)
            test_dir = '/mnt/dataset1/plzhang/data/'+test_dir+'/pytorch'
            # test_dir = '/data/plzhang/data/'+test_dir+'/pytorch'
            # test_dir = 'data/'+test_dir+'/pytorch'

            dataset = test_dir.split('/')[1]
            gallery_dataset = datasets.ImageFolder(os.path.join(test_dir, 'gallery'))
            query_dataset = datasets.ImageFolder(os.path.join(test_dir, 'query'))

            gallery_dataset = ImageDataset(gallery_dataset.imgs, self.data_transforms['val'])
            query_dataset = ImageDataset(query_dataset.imgs, self.data_transforms['val'])
            self.test_loaders[dataset_name] = {key: torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=self.batch_size,
                                                shuffle=False, 
                                                # num_workers=0, 
                                                num_workers=0, 
                                                pin_memory=True) for key, dataset in {'gallery': gallery_dataset, 'query': query_dataset}.items()}
        

            gallery_cameras, gallery_labels = get_camera_ids(gallery_dataset.imgs)
            self.gallery_meta[dataset_name] = {
            # self.gallery_meta[dataset] = {
                'sizes':  len(gallery_dataset),
                'cameras': gallery_cameras,
                'labels': gallery_labels
            }

            query_cameras, query_labels = get_camera_ids(query_dataset.imgs)
            self.query_meta[dataset_name] = {
                'sizes':  len(query_dataset),
                'cameras': query_cameras,
                'labels': query_labels
            }

        print('Query Sizes:', self.query_meta[dataset_name]['sizes'])
        print('Gallery Sizes:', self.gallery_meta[dataset_name]['sizes'])
        # print('Query Sizes:', self.query_meta[dataset]['sizes'])
        # print('Gallery Sizes:', self.gallery_meta[dataset]['sizes'])

    def preprocess(self):
        self.transform()
        self.preprocess_train()
        self.preprocess_test()

def get_camera_ids(img_paths):
    """get camera id and labels by image path
    """
    camera_ids = []
    labels = []
    for path, v in img_paths:
        filename = os.path.basename(path)
        # print('filename', filename)
        
        # if filename[:3]!='cam':
        if 'c' not in filename:
            label = filename[0:4]
            camera = filename.split('_')[2]
        elif filename[:3]!='cam':
            # print('filename', filename[:3])
            label = filename[0:4]
            camera = filename.split('c')[1]
            camera = camera.split('s')[0]
        
        else:
            label = filename.split('_')[2]
            camera = filename.split('_')[1]
        
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_ids.append(int(camera[0]))
    return camera_ids, labels



def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids



if __name__ == '__main__':
    test()
