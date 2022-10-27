import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import pprint
from loguru import logger

from utils import config, logger_tools, other_tools
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func


class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
    
    def train(self, epoch, tf_writter):
        use_adv = bool(epoch>=self.no_adv_epochs)
        self.model.train()
        self.d_model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        for its, (tar_pose, in_audio, in_facial, in_word, in_id, in_emo, in_sem) in enumerate(self.train_loader):
#             if its+1 == its_len and tar_pose.shape[0] < self.batchnorm_bug: # skip final bs=1, bug for bn
#                     continue
            t_data = time.time() - t_start
           
            tar_pose = tar_pose.cuda()
            in_audio = in_audio.cuda() if self.audio_rep is  not "None" else None  
            in_facial = in_facial.cuda() if self.facial_rep is  not "None" else None
            in_id = in_id.cuda() if self.facial_rep is  not "None" else None
            in_word = in_word.cuda() if self.word_rep is  not "None" else None
            in_emo = in_emo.cuda() if self.emo_rep is  not "None" else None
            in_sem = in_sem.cuda() if self.sem_rep is  not "None" else None
            
            in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
            in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
            in_pre_pose[:, 0:self.pre_frames, -1] = 1 
        
            t_data = time.time() - t_start 
            
            # --------------------------- d training --------------------------------- #
            d_loss_final = 0
            if use_adv:
                self.opt_d.zero_grad()
                out_pose  = self.model(in_pre_pose, in_audio=in_audio, in_facial=in_facial, in_text=in_word, in_id=in_id, in_emo=in_emo)
                out_d_fake = self.d_model(out_pose)
                # d_fake_for_d = self.adv_loss(out_d_fake, fake_gt)
                out_d_real = self.d_model(tar_pose)
                # d_real_for_d = self.adv_loss(out_d_real, real_gt)
                d_loss_adv = torch.sum(-torch.mean(torch.log(out_d_real + 1e-8) + torch.log(1 - out_d_fake + 1e-8)))
                d_loss_final += d_loss_adv
                self.loss_meters[3].update(d_loss_final.item()) # we ignore batch_size here
                d_loss_final.backward()
                self.opt_d.step()
                # if lrs_d is not None: lrs_d.step()       
            self.opt.zero_grad()

 
            # --------------------------- g training --------------------------------- #
            g_loss_final = 0
            out_pose  = self.model(in_pre_pose, in_audio=in_audio, in_facial=in_facial, in_text=in_word, in_id=in_id, in_emo=in_emo)
            use_sem_weight = True
            #print(tar_pose.shape, in_sem.shape)
            if use_sem_weight:
                huber_value = self.rec_loss(tar_pose*(in_sem.unsqueeze(2)+1), out_pose*(in_sem.unsqueeze(2)+1))
            else: huber_value = self.rec_loss(tar_pose, out_pose)
            huber_value *= self.rec_weight 
            self.loss_meters[1].update(huber_value.item())
            g_loss_final += huber_value 
            if use_adv:
                dis_out = self.d_model(out_pose)
                d_fake_value = -torch.mean(torch.log(dis_out + 1e-8)) # self.adv_loss(out_d_fake, real_gt) # here 1 is real
                d_fake_value *= self.adv_weight * d_fake_value
                self.loss_meters[2].update(d_fake_value.item())
                g_loss_final += d_fake_value
                
#                 latent_out = self.eval_model(out_pose)
#                 latent_ori = self.eval_model(tar_pose)
#                 huber_fid_loss = self.rec_loss(latent_out, latent_ori) * self.fid_weight
#                 self.loss_meters[4].update(huber_fid_loss.item())
#                 g_loss_final += huber_fid_loss
            
            self.loss_meters[0].update(g_loss_final.item())
            g_loss_final.backward()
            if self.grad_norm != 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.opt.step()
            # if lrs is not None: lrs.step() 
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            lr_d = self.opt_d.param_groups[0]['lr']
            
            # --------------------------- recording ---------------------------------- #
            if its % self.log_period == 0:
                pstr = "[%d][%d/%d]\t"%(epoch, its, its_len)
                for loss_meter in self.loss_meters:
                    if loss_meter.count > 0:
                        pstr += "{}: {:.3f}\t".format(loss_meter.name, loss_meter.avg)
                        tf_writter.add_scalar(loss_meter.name, loss_meter.avg, its)
                        loss_meter.reset()
                pstr += "data: %d ms\t"%(t_data*1000)        
                pstr += "net: %d ms\t"%(t_train*1000)
                pstr += "lr: {:.1e}\t".format(lr_g)
                pstr += "dlr: {:.1e}\t".format(lr_d)
                #pstr += "mem: {:.2f}Gb".format(mem_cost)
                logger.info(pstr)
                
    
    def val_fid(self, epoch, tf_writter):
        self.model.eval()
        with torch.no_grad():
            its_len = len(self.val_loader)
            for its, (tar_pose, in_audio, in_facial, in_word, in_id, in_emo, in_sem) in enumerate(self.val_loader):
#                 if its+1 == its_len and tar_pose.shape[0] < self.batchnorm_bug: # skip final bs=1, bug for bn
#                     continue
                tar_pose = tar_pose.cuda()
                in_audio = in_audio.cuda() if self.audio_rep is  not "None" else None  
                in_facial = in_facial.cuda() if self.facial_rep is  not "None" else None
                in_id = in_id.cuda() if self.facial_rep is  not "None" else None
                in_word = in_word.cuda() if self.word_rep is  not "None" else None
                in_emo = in_emo.cuda() if self.emo_rep is  not "None" else None

                in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                in_pre_pose[:, 0:self.pre_frames, -1] = 1  # indicating bit for constraints
    
                out_pose = self.model(in_pre_pose, in_audio=in_audio, in_facial=in_facial, in_text=in_word, in_id=in_id, in_emo=in_emo)
                latent_out = self.eval_model(out_pose)
                latent_ori = self.eval_model(tar_pose)
                #print(latent_out,latent_ori)
                if its == 0:
                    latent_out_all = latent_out.cpu().numpy()
                    latent_ori_all = latent_ori.cpu().numpy()
                else:
                    latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()], axis=0)
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()], axis=0)
                huber_value = self.rec_loss(tar_pose, out_pose)
                huber_value *= self.rec_weight
            fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        tf_writter.add_scalar("huber_value", huber_value, epoch)
        tf_writter.add_scalar("fid", fid, epoch)
        return huber_value, fid 
                
        
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        self.model.eval()
        with torch.no_grad():
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            for its, (tar_pose, in_audio, in_facial, in_word, in_id, in_emo, in_sem) in enumerate(self.test_loader):
                # tar_pose = tar_pose.cuda() # no mean
                in_audio = in_audio.cuda() if self.audio_rep is  not "None" else None  
                in_facial = in_facial.cuda() if self.facial_rep is  not "None" else None
                in_id = in_id.cuda() if self.facial_rep is  not "None" else None
                in_word = in_word.cuda() if self.word_rep is  not "None" else None
                in_emo = in_emo.cuda() if self.emo_rep is  not "None" else None
                
                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1
                
                in_audio = in_audio.reshape(1, -1)   
                out_dir_vec = self.model(**dict(pre_seq=pre_pose, in_audio=in_audio, in_text=in_word, in_facial=in_facial, in_id=in_id, in_emo=in_emo))
                out_final = (out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                #out_final = out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) + self.mean_pose
                total_length += out_final.shape[0]
                #print(out_final.shape)

                with open(f"{results_save_path}result_raw_{self.vis_lookuptable[its]}.bvh", 'w+') as f_real:
                    for line_id in range(out_final.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')                    
        data_tools.result2target_vis(self.pose_rep, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")
    