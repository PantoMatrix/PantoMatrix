import train
import os
import time
import csv
import sys
import warnings
import random
import numpy as np
import time
import pprint
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import smplx

from utils import config, logger_tools, other_tools, metric
from utils import rotation_conversions as rc
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from scipy.spatial.transform import Rotation


class CustomTrainer(train.BaseTrainer):
    """
    motion representation learning
    """
    def __init__(self, args):
        super().__init__(args)

        ##--------------Copy from BEAT2022, ae_trainer.py------------##
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
        self.rec_weight = args.rec_weight
        self.vel_weight = args.vel_weight
        self.args = args

        # self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        # self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
        
        # self.audio_norm = args.audio_norm
        # self.facial_norm = args.facial_norm
        # if self.audio_norm:
        #     self.mean_audio = np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_mean.npy")
        #     self.std_audio = np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_std.npy")
        # if self.facial_norm:
        #     self.mean_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")
        #     self.std_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy")

        # self.ori_joint_list = joints_list[args.ori_joints]
        # self.tar_joint_list = joints_list[args.tar_joints]
        
        ##--------------Copy from BEAT2022, ae_trainer.py------------##
    
    def train(self, epoch):
        self.model.train()
        
        ##--------------Copy from BEAT2022, ae_trainer.py------------##
        its_len = len(self.train_loader)
        ##--------------Copy from BEAT2022, ae_trainer.py------------##

        t_start = time.time()

        for its, dict_data in enumerate(self.train_loader):
            tar_pose = dict_data["pose"]
            
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            t_data = time.time() - t_start
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            
            tar_pose = tar_pose.cuda()  

            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            t_data = time.time() - t_start
            
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            
            
            self.opt.zero_grad()
           
            net_out = self.model(tar_pose)
            rec_pose = net_out["rec_pose"]
            ##-----------------Reconstruction loss------------##
            recon_loss = self.rec_loss(rec_pose, tar_pose)
         
            recon_loss = torch.mean(recon_loss, dim=(1, 2))
            self.loss_meters['rec_l1'].update(torch.sum(recon_loss).item()*self.rec_weight)
            recon_loss = torch.sum(recon_loss*self.rec_weight)
            ##-----------------Reconstruction loss------------##

            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            ## Cần thêm cel_weight vào file config
            # rec vel loss
            if self.vel_weight > 0:  # use pose diff
                target_diff = tar_pose[:, 1:] - tar_pose[:, :-1]
                recon_diff = rec_pose[:, 1:] - rec_pose[:, :-1]
                vel_rec_loss = torch.mean(self.vel_loss(recon_diff, target_diff), dim=(1, 2))
                self.loss_meters['vel_l1'].update(torch.sum(vel_rec_loss).item()*self.vel_weight)
                recon_loss += (torch.sum(vel_rec_loss)*self.vel_weight)

            loss = recon_loss
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            
            
            # ---------------------- vae -------------------------- #
            if "VQVAE" in self.args.g_name:
                loss_embedding = net_out["embedding_loss"]
                loss += loss_embedding

            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            self.loss_meters['all'].update(loss.item())
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            
            loss.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.recording(epoch, its, its_len, self.loss_meters, lr_g, 0, t_data, t_train, mem_cost)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        t_start = time.time()
        with torch.no_grad():
            its_len = len(self.val_loader)
            for its, dict_data in enumerate(self.val_loader):
                tar_pose = dict_data["pose"]
                tar_pose = tar_pose.cuda()         
                t_data = time.time() - t_start 

                #self.opt.zero_grad()
                #g_loss_final = 0
                net_out = self.model(tar_pose)
                rec_pose = net_out["rec_pose"]
                
                ##--------------Copy from BEAT2022, ae_trainer.py------------##
                if self.vel_weight > 0:  # use pose diff
                    target_diff = tar_pose[:, 1:] - tar_pose[:, :-1]
                    recon_diff = rec_pose[:, 1:] - rec_pose[:, :-1]
                    vel_rec_loss = torch.mean(self.vel_loss(recon_diff, target_diff), dim=(0, 1, 2))
                    self.loss_meters['vel_val'].update(vel_rec_loss.item())    
                recon_loss = F.l1_loss(rec_pose, tar_pose, reduction='none')
                recon_loss = torch.mean(recon_loss, dim=(0, 1, 2))
                loss = recon_loss
                ##--------------Copy from BEAT2022, ae_trainer.py------------##

           
                if "VQVAE" in self.args.g_name:
                    loss_embedding = net_out["embedding_loss"]
                    loss += loss_embedding
                    
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            self.loss_meters['rec_val'].update(loss.item())
            self.val_recording(epoch, self.loss_meters)
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0

        ##-------------------Copy from BEAT2022, ae_trainer.py------------##
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        ##-------------------Copy from BEAT2022, ae_trainer.py------------##
        self.model.eval()
        
        with torch.no_grad():
            
            ##-------------------Copy from BEAT2022, ae_trainer.py------------##
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            ##-------------------Copy from BEAT2022, ae_trainer.py------------##
            
            for its, dict_data in enumerate(self.test_loader):
                tar_pose = dict_data["pose"]
                tar_pose = tar_pose.cuda()

                for i in range(tar_pose.shape[1]//(self.pose_length)):
                    tar_pose_new = tar_pose[:,i*(self.pose_length):i*(self.pose_length)+self.pose_length,:]
                    # print(tar_pose_new.shape)
                    recon_data = self.model(tar_pose_new)

                    std_pose = self.test_data.std_pose[self.test_data.joint_mask.astype(bool)]
                    mean_pose = self.test_data.mean_pose[self.test_data.joint_mask.astype(bool)]
                    out_sub = (recon_data['rec_pose'].cpu().numpy().reshape(-1, self.args.pose_dims) * std_pose) + mean_pose
                    out_final = out_sub

                total_length += out_final.shape[0]
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(out_final.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')
                        
            data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
            
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")