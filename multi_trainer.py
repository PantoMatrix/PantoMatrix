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
    '''
    SIGGRAPH ASIA 2020
    '''
    def __init__(self, args):
        super().__init__(args)
        self.word_rep = args.word_rep
        self.speaker_id = args.speaker_id
        self.rec_loss = get_loss_func("huber_loss", beta=0.1)
        self.rec_loss_rand = get_loss_func("huber_loss", beta=0.05, reduction="none")
        if self.ddp: 
            self.rec_loss.to(self.rank)
            self.rec_loss_rand.to(self.rank)
            
        self.div_reg_weight = args.div_reg_weight
        self.kld_weight = args.kld_weight
        self.loss_meters = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'),
            'reg': other_tools.AverageMeter('reg'),
            'kld': other_tools.AverageMeter('kld'),
        } 
        
    def train(self, epoch):
        use_adv = bool(epoch>=self.no_adv_epochs)
        self.model.train()
        self.d_model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        # using facial or text here
        for its,  dict_data in enumerate(self.train_loader):
            tar_pose, in_audio, in_word, in_id = dict_data["pose"], dict_data["audio"], dict_data["word"], dict_data["id"]
            t_data = time.time() - t_start
            if self.ddp:
                tar_pose = tar_pose.to(self.rank)
                in_audio = in_audio.to(self.rank) if self.audio_rep is not None else None
                in_word = in_word.to(self.rank) if self.word_rep is not None else None
                in_id = in_id.long().to(self.rank) if self.speaker_id else None
                in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).to(self.rank)
            else:
                tar_pose = tar_pose.cuda()
                in_audio = in_audio.cuda() if self.audio_rep is not None else None
                in_word = in_word.cuda() if self.word_rep is not None else None
                in_id = in_id.long().cuda() if self.speaker_id else None
                in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
            
            in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
            in_pre_pose[:, 0:self.pre_frames, -1] = 1 
            t_data = time.time() - t_start 
            
            # --------------------------- d training --------------------------------- #
            d_loss_final = 0
            out_pose, z, z_mu, z_logvar = self.model(**dict(pre_seq=in_pre_pose, in_audio=in_audio, in_word=in_word, in_id=in_id))
            #print(z,z_mu)
            if use_adv:
                self.opt_d.zero_grad()
                #out_pose, _, _, _ = self.model(in_pre_pose, in_audio=in_audio, in_facial=in_facial, in_id=in_id)
                out_d_fake = self.d_model(**dict(poses=out_pose))
                # d_fake_for_d = self.adv_loss(out_d_fake, fake_gt)
                out_d_real = self.d_model(**dict(poses=tar_pose))
                # d_real_for_d = self.adv_loss(out_d_real, real_gt)
                d_loss_adv = torch.sum(-torch.mean(torch.log(out_d_real + 1e-8) + torch.log(1 - out_d_fake + 1e-8)))
                d_loss_final += d_loss_adv
                self.loss_meters['dis'].update(d_loss_final.item()) # we ignore batch_size here
                d_loss_final.backward(retain_graph=True)
                self.opt_d.step()
                # if lrs_d is not None: lrs_d.step()       
            self.opt.zero_grad()

 
            # --------------------------- g training --------------------------------- #
            g_loss_final = 0
            #out_pose, z, z_mu, z_logvar = self.model(in_pre_pose, in_audio=in_audio, in_facial=in_facial, in_id=in_id)
            huber_value = self.rec_loss(tar_pose, out_pose)
            huber_value *= self.rec_weight
            self.loss_meters['rec'].update(huber_value.item())
            g_loss_final += huber_value 
            if use_adv:
                dis_out = self.d_model(out_pose)
                d_fake_value = -torch.mean(torch.log(dis_out + 1e-8)) # self.adv_loss(out_d_fake, real_gt) # here 1 is real
                d_fake_value *= self.adv_weight
                self.loss_meters['gen'].update(d_fake_value.item())
                g_loss_final += d_fake_value
                
            if self.speaker_id:
                rand_idx = torch.randperm(in_id.shape[0])
                rand_vids = in_id[rand_idx]
                out_pose_rand_vid, z_rand_vid, _, _ = self.model(**dict(pre_seq=in_pre_pose, in_audio=in_audio, in_word=in_word, in_id=rand_vids))
                huber_value_rand = self.rec_loss_rand(out_pose_rand_vid, out_pose)
                #print(self.rec_loss_rand.reduction, self.rec_loss_rand.beta)
                huber_value_rand = huber_value_rand.sum(dim=1).sum(dim=1)
                huber_value_rand = huber_value_rand.view(huber_value_rand.shape[0], -1).mean(1)
                z_l1 = F.l1_loss(z.detach(), z_rand_vid.detach(), reduction='none')
                z_l1 = z_l1.view(z_l1.shape[0], -1).mean(1)
                div_reg = -(huber_value_rand / (z_l1 + 1.0e-5))
                div_reg = torch.clamp(div_reg, min=-1000)
                div_reg = div_reg.mean() * self.div_reg_weight
                g_loss_final += div_reg
                self.loss_meters['reg'].update(div_reg.item())
                kld = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
                kld = kld * self.kld_weight
                g_loss_final += kld
                self.loss_meters['kld'].update(kld.item())
            
            self.loss_meters['all'].update(g_loss_final.item())
            g_loss_final.backward()
            #if self.grad_norm != 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.opt.step()
            # if lrs is not None: lrs.step() 
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            lr_d = self.opt_d.param_groups[0]['lr']
            
            # --------------------------- recording ---------------------------------- #
            if its % self.log_period == 0:
                 self.recording(epoch, its, its_len, self.loss_meters, lr_g, lr_d, t_data, t_train, mem_cost)
            #if its == 1:break
        self.opt_s.step(epoch)
        self.opt_d_s.step(epoch) 
        
    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            its_len = len(self.val_loader)
            for its, dict_data in enumerate(self.val_loader):
                tar_pose, in_audio, in_word, in_id = dict_data["pose"], dict_data["audio"], dict_data["word"], dict_data["id"]
                if self.ddp:
                    tar_pose = tar_pose.to(self.rank)
                    in_audio = in_audio.to(self.rank) if self.audio_rep is not None else None
                    in_word = in_word.to(self.rank) if self.word_rep is not None else None
                    in_id = in_id.long().to(self.rank) if self.speaker_id else None
                    in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).to(self.rank)
                else:
                    tar_pose = tar_pose.cuda()
                    in_audio = in_audio.cuda() if self.audio_rep is not None else None
                    in_word = in_word.cuda() if self.word_rep is not None else None
                    in_id = in_id.long().cuda() if self.speaker_id else None
                    in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                    
                in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                in_pre_pose[:, 0:self.pre_frames, -1] = 1  # indicating bit for constraints
                out_pose, _, _, _ = self.model(**dict(pre_seq=in_pre_pose, in_audio=in_audio, in_word=in_word, in_id=in_id))
                #print(out_pose.shape, tar_pose.shape)
                latent_out = self.eval_model(out_pose)
                latent_ori = self.eval_model(tar_pose)
                #print(latent_out,latent_ori)
                if its == 0:
                    latent_out_all = latent_out.detach().cpu().numpy()
                    latent_ori_all = latent_ori.detach().cpu().numpy()
                else:
                    latent_out_all = np.concatenate([latent_out_all, latent_out.detach().cpu().numpy()], axis=0)
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori.detach().cpu().numpy()], axis=0)
                huber_value = self.rec_loss(tar_pose, out_pose)
                huber_value *= self.rec_weight
                self.loss_meters['rec_val'].update(huber_value.item())
                #if its == 1:break
            fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
            self.loss_meters['fid_val'].update(fid)
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
                tar_pose, in_audio, in_word, in_id = dict_data["pose"], dict_data["audio"], dict_data["word"], dict_data["id"]
                n_audio = in_audio.cuda()  
                in_word = in_word.cuda() if self.word_rep is not None else None
                in_id = in_id.long().cuda() if self.speaker_id else None
                
                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1
                
                in_audio = in_audio.reshape(1, -1)   
                out_dir_vec, *_ = self.model(**dict(pre_seq=pre_pose, in_audio=in_audio, in_word=in_word, in_id=in_id))
                out_final = (out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                total_length += out_final.shape[0]
  
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(out_final.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')

            data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
            end_time = time.time() - start_time
            logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")
