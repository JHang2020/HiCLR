import sys
import argparse
import yaml
import math
import random
import numpy as np
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from .utils import *
# torchlight
import torchlight.torchlight as torchlight
from torchlight.torchlight import str2bool
from torchlight.torchlight import DictAction
from torchlight.torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class HiCLR_Processor(PT_Processor):
    """
        Processor for HiCLR Pre-training.
    """
    def __init__(self, argv=None):
        super().__init__(argv)
        self.mask_gen = Jointmask(**(self.arg.mask_args))

    def train(self, epoch):
        self.model.train()
        self.ep = epoch
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_value1 = []
        loss_value2 = []

        for [data1, data2, data3, data4], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            data3 = data3.float().to(self.dev, non_blocking=True)
            data4 = data4.float().to(self.dev, non_blocking=True)
            #data5 = data5.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            if self.arg.stream == 'joint':
                pass
            elif self.arg.stream == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)
                motion3 = torch.zeros_like(data3)
                motion4 = torch.zeros_like(data4)
                #motion5 = torch.zeros_like(data5)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]
                motion3[:, :, :-1, :, :] = data3[:, :, 1:, :, :] - data3[:, :, :-1, :, :]
                motion4[:, :, :-1, :, :] = data4[:, :, 1:, :, :] - data4[:, :, :-1, :, :]
                #motion5[:, :, :-1, :, :] = data5[:, :, 1:, :, :] - data5[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
                data3 = motion3
                data4 = motion4
                #data5 = motion5
            elif self.arg.stream == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)
                bone3 = torch.zeros_like(data3)
                bone4 = torch.zeros_like(data4)
                #bone5 = torch.zeros_like(data5)
                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]
                    bone3[:, :, :, v1 - 1, :] = data3[:, :, :, v1 - 1, :] - data3[:, :, :, v2 - 1, :]
                    bone4[:, :, :, v1 - 1, :] = data4[:, :, :, v1 - 1, :] - data4[:, :, :, v2 - 1, :]
                    #bone5[:, :, :, v1 - 1, :] = data5[:, :, :, v1 - 1, :] - data5[:, :, :, v2 - 1, :]

                data1 = bone1
                data2 = bone2
                data3 = bone3
                data4 = bone4
                #data5 = bone5
            else:
                raise ValueError

            # forward
            #NOTE: data2 data3 is normal aug, data1 is the strong aug of the data2 !
            if epoch <= self.arg.mining_epoch or 'skeletonclr' in self.arg.exp_descri:
                if self.arg.exp_descri == 'MutalDDM_with4parallel_ablmask':
                    m = self.mask_gen(data1,data1.shape[2]).to(data1)#N,C,T,V,M
                    output1, target1, output2, output3, target2, target3 = self.model.forward_pretrain_wmask(im_q_1=data3, im_q_2=data4,im_q=data1, im_k=data2, mask=m)
                    if hasattr(self.model, 'module'):
                        self.model.module.update_ptr(output1.size(0))
                    else:
                        self.model.update_ptr(output1.size(0))
                    loss1 = self.loss(output1, target1)
                    loss2 = -torch.mean(torch.sum(torch.log(output2) * target2, dim=1))  # DDM loss
                    loss3 = -torch.mean(torch.sum(torch.log(output3) * target3, dim=1))  # DDM loss
                    loss = loss1 + (loss2 + loss3) / 2.
            else:
                if self.arg.exp_descri == 'MutalDDM_with4parallel_ablmask':
                    m = self.mask_gen(data1,data1.shape[2]).to(data1)#N,C,T,V,M
                    output1, mask, output2, output3, target2, target3 = self.model.forward_pretrain_wmask(im_q_1=data3, im_q_2=data4,im_q=data1, im_k=data2, nnm=True, topk=self.arg.topk,mask=m)
                    if hasattr(self.model, 'module'):
                        self.model.module.update_ptr(output1.size(0))
                    else:
                        self.model.update_ptr(output1.size(0))
                    loss1 = - (F.log_softmax(output1, dim=1) * mask).sum(1) / mask.sum(1)
                    loss1 = loss1.mean()
                    loss2 = -torch.mean(torch.sum(torch.log(output2) * target2, dim=1))  # DDM loss
                    loss3 = -torch.mean(torch.sum(torch.log(output3) * target3, dim=1))  # DDM loss
                    loss = loss1 + (loss2 + loss3) / 2.

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss1'] = loss2.data.item()
            self.iter_info['loss2'] = loss3.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            loss_value1.append(loss2.data.item())
            loss_value2.append(loss3.data.item())

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss1', np.mean(loss_value1), epoch)
        self.train_writer.add_scalar('loss2', np.mean(loss_value2), epoch)

        self.show_epoch_info()

    @torch.no_grad()
    def knn_monitor(self, epoch):
        if len(self.gpus) > 1:
            self.model.module.encoder_q.eval()
        else:
            self.model.encoder_q.eval()
        feature_bank, label_bank = [], []
        with torch.no_grad():
            # generate feature bank
            for data, label in tqdm(self.data_loader['mem_train'], desc='Feature extracting'):
                data = data.float().to(self.dev, non_blocking=True)
                label = label.long().to(self.dev, non_blocking=True)

                #data = self.view_gen(data)
                if len(self.gpus) > 1:
                    feature = self.model.module.encoder_q(data)
                else:
                    feature = self.model.encoder_q(data)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                label_bank.append(label)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.cat(label_bank).to(feature_bank.device)
            # loop test data to predict the label by weighted knn search
            for i in self.arg.knn_k:
                total_top1, total_top5, total_num = 0, 0, 0
                test_bar = tqdm(self.data_loader['mem_test'], desc='kNN-{}'.format(i))
                for data, label in test_bar:
                    data = data.float().to(self.dev, non_blocking=True)
                    label = label.float().to(self.dev, non_blocking=True)

                    #data = self.view_gen(data)

                    if len(self.gpus) > 1:
                        feature = self.model.module.encoder_q(data)
                    else:
                        feature = self.model.encoder_q(data)
                    feature = F.normalize(feature, dim=1)

                    pred_labels = knn_predict(feature, feature_bank, feature_labels, self.arg.knn_classes, i,
                                              self.arg.knn_t)

                    total_num += data.size(0)
                    total_top1 += (pred_labels[:, 0] == label).float().sum().item()
                    test_bar.set_postfix({'k': i, 'Accuracy': total_top1 / total_num * 100})
                acc = total_top1 / total_num * 100

                self.knn_results[i][epoch] = acc

                self.train_writer.add_scalar('KNN-{}'.format(i), acc, epoch)
    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')
        parser.add_argument('--warmup_epochs', type=int, default=0, help='topk samples in nearest neighbor mining')
        parser.add_argument('--mask_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--exp_descri', type=str, default='StandardDDM', help='Describe the method')
        parser.add_argument('--randenque', type=str2bool, default=False, help='random enqueue or not')
        parser.add_argument('--weight1', type=float, default=0.5, help='weight for branch1')
        parser.add_argument('--weight2', type=float, default=0.5, help='weight for branch2')
        return parser
