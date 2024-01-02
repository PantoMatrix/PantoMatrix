# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Train script for audio2pose

import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger

from utils import config, logger_tools, other_tools
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from train import BaseTrainer


class CustomTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.g_name = args.g_name
        self.pose_length = args.pose_length
        self.loss_meters = {
            'rec_val': other_tools.AverageMeter('rec_val'),
            'vel_val': other_tools.AverageMeter('vel_val'),
            'kl_val': other_tools.AverageMeter('kl_val'),
            'all': other_tools.AverageMeter('all'),
            'rec_l1': other_tools.AverageMeter('rec_l1'), 
            'vel_l1': other_tools.AverageMeter('vel_l1'),
            'kl_loss': other_tools.AverageMeter('kl_loss'),
            #'acceleration_loss': other_tools.AverageMeter('acceleration_loss'),
        }
        self.best_epochs = {
            'rec_val': [np.inf, 0],
            'vel_val': [np.inf, 0],
            'kl_val': [np.inf, 0],
                           }
        self.rec_loss = torch.nn.L1Loss(reduction='none')
        self.vel_loss = torch.nn.MSELoss(reduction='none')
        self.variational_encoding = args.variational_encoding
        self.rec_weight = args.rec_weight
        self.vel_weight = args.vel_weight
    
    def train(self, epoch):
        self.model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        
        for its, dict_data in enumerate(self.train_loader):
            tar_pose = dict_data["pose"]
            t_data = time.time() - t_start
            tar_pose = tar_pose.cuda()
            t_data = time.time() - t_start 

            self.opt.zero_grad()
            poses_feat, pose_mu, pose_logvar, recon_data = \
                self.model(None, tar_pose, variational_encoding=self.variational_encoding)
            
            recon_loss = self.rec_loss(recon_data, tar_pose) # 128*34*123
            recon_loss = torch.mean(recon_loss, dim=(1, 2)) # 128
            self.loss_meters['rec_l1'].update(torch.sum(recon_loss).item()*self.rec_weight)
            recon_loss = torch.sum(recon_loss*self.rec_weight)
            # rec vel loss
            if self.vel_weight > 0:  # use pose diff
                target_diff = tar_pose[:, 1:] - tar_pose[:, :-1]
                recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
                vel_rec_loss = torch.mean(self.vel_loss(recon_diff, target_diff), dim=(1, 2))
                self.loss_meters['vel_l1'].update(torch.sum(vel_rec_loss).item()*self.vel_weight)
                recon_loss += (torch.sum(vel_rec_loss)*self.vel_weight)
            # KLD
            if self.variational_encoding:
                KLD = -0.5 * torch.sum(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
                if epoch < 10:
                    KLD_weight = 0
                else:
                    KLD_weight = min(1.0, (epoch - 10) * 0.05) * 0.01
                loss = recon_loss + KLD_weight * KLD
                self.loss_meters['kl_loss'].update(KLD_weight * KLD.item())
            else:
                loss = recon_loss
            self.loss_meters['all'].update(loss.item())
            if self.grad_norm != 0 and "LSTM" in self.g_name: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
#             logger.warning(total_norm)
            loss.backward()
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # --------------------------- recording ---------------------------------- #
            if its % self.log_period == 0:
                self.recording(epoch, its, its_len, self.loss_meters, lr_g, 0, t_data, t_train, mem_cost)   
            #if its == 1:break
        self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            its_len = len(self.val_loader)
            for its, dict_data in enumerate(self.val_loader):
                tar_pose = dict_data["pose"]
                tar_pose = tar_pose.cuda()
                
                poses_feat, pose_mu, pose_logvar, recon_data = \
                self.model(None, tar_pose, variational_encoding=self.variational_encoding)
                if self.vel_weight > 0:  # use pose diff
                    target_diff = tar_pose[:, 1:] - tar_pose[:, :-1]
                    recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
                    vel_rec_loss = torch.mean(self.vel_loss(recon_diff, target_diff), dim=(0, 1, 2))
                    self.loss_meters['vel_val'].update(vel_rec_loss.item())
                #print(recon_data.shape, tar_pose.shape)    
                recon_loss = F.l1_loss(recon_data, tar_pose, reduction='none')
                recon_loss = torch.mean(recon_loss, dim=(0, 1, 2))
                self.loss_meters['rec_val'].update(recon_loss.item())
            self.val_recording(epoch, self.loss_meters)
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        self.model.eval()
        with torch.no_grad():
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            for its, dict_data in enumerate(self.test_loader):
                tar_pose = dict_data["pose"]
                tar_pose = tar_pose.cuda() # no mean
                if "LSTM" in self.g_name or "multi_length" in self.notes:
                    poses_feat, pose_mu, pose_logvar, recon_data = \
                    self.model(**dict(pre_poses=None, poses=tar_pose, variational_encoding=self.variational_encoding))
                    out_final = (recon_data.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose                  
                else:
                    for i in range(tar_pose.shape[1]//(self.pose_length)):
                        tar_pose_new = tar_pose[:,i*(self.pose_length):i*(self.pose_length)+self.pose_length,:]
                        poses_feat, pose_mu, pose_logvar, recon_data = \
                    self.model(**dict(pre_poses=None, poses=tar_pose_new, variational_encoding=self.variational_encoding,))
                        out_sub = (recon_data.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                        if i != 0:
                            out_final = np.concatenate((out_final,out_sub), 0)
                        else:
                            out_final = out_sub
                
                total_length += out_final.shape[0]
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(out_final.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')
            data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
            end_time = time.time() - start_time
            logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")