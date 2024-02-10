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
import librosa

from utils import config, logger_tools, other_tools, metric
from utils import rotation_conversions as rc
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from scipy.spatial.transform import Rotation


class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.joints = self.train_data.joints
        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'div_reg', "kl"], [False,True,True, False, False, False, False, False, False, False, False, False, False])
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)

    def _load_data(self, dict_data):
        tar_pose = dict_data["pose"].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        tar_word = dict_data["word"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank)
        in_emo = dict_data["emo"].to(self.rank) 
        #in_sem = dict_data["sem"].to(self.rank) 
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        in_pre_pose_cat = torch.cat([tar_pose[:, 0:self.args.pre_frames], tar_trans[:, :self.args.pre_frames]], dim=2).to(self.rank)

        in_pre_pose = tar_pose.new_zeros((bs, n, j*6+1+3)).to(self.rank)
        in_pre_pose[:, 0:self.args.pre_frames, :-1] = in_pre_pose_cat[:, 0:self.args.pre_frames]
        in_pre_pose[:, 0:self.args.pre_frames, -1] = 1
        return {
            "tar_pose": tar_pose,
            "in_audio": in_audio,
            "in_motion": in_pre_pose,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_word": tar_word,
            'tar_id': tar_id,
            'in_emo': in_emo,
            #'in_sem': in_sem,
        }
    
    def _d_training(self, loaded_data):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        net_out  = self.model(in_audio = loaded_data['in_audio'], pre_seq = loaded_data["in_motion"], in_text=loaded_data["tar_word"], in_id=loaded_data["tar_id"], in_emo=loaded_data["in_emo"], in_facial = loaded_data["tar_exps"])
        rec_pose = net_out["rec_pose"][:, :, :j*6]
        # rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3]
        
        rec_pose = rec_pose.reshape(bs, n, j, 6)
        rec_pose = rc.rotation_6d_to_matrix(rec_pose)
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.rotation_6d_to_matrix(loaded_data["tar_pose"].reshape(bs, n, j, 6))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        out_d_fake = self.d_model(rec_pose)
        out_d_real = self.d_model(tar_pose)
        
        d_loss_adv = torch.sum(-torch.mean(torch.log(out_d_real + 1e-8) + torch.log(1 - out_d_fake + 1e-8)))
        self.tracker.update_meter("dis", "train", d_loss_adv.item())
        return d_loss_adv
    
    def _g_training(self, loaded_data, use_adv, mode="train"):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        net_out  = self.model(in_audio = loaded_data['in_audio'], pre_seq = loaded_data["in_motion"], in_text=loaded_data["tar_word"], in_id=loaded_data["tar_id"], in_emo=loaded_data["in_emo"], in_facial = loaded_data["tar_exps"])
        rec_pose = net_out["rec_pose"][:, :, :j*6]
        rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3]
        # print(rec_pose.shape, bs, n, j, loaded_data['in_audio'].shape, loaded_data["in_motion"].shape)
        rec_pose = rec_pose.reshape(bs, n, j, 6)
        rec_pose = rc.rotation_6d_to_matrix(rec_pose)
        tar_pose = rc.rotation_6d_to_matrix(loaded_data["tar_pose"].reshape(bs, n, j, 6))

        rec_loss = self.rec_loss(tar_pose, rec_pose)
        rec_loss *= self.args.rec_weight
        self.tracker.update_meter("rec", mode, rec_loss.item())
        # rec_loss_vel = self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1])
        # self.tracker.update_meter("vel", mode, rec_loss_vel.item())
        # rec_loss_acc = self.vel_loss(rec_pose[:, 2:] - 2*rec_pose[:, 1:-1] + rec_pose[:, :-2], tar_pose[:, 2:] - 2*tar_pose[:, 1:-1] + tar_pose[:, :-2])
        # self.tracker.update_meter("acc", mode, rec_loss_acc.item())
        
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        if self.args.pose_dims < 330 and mode != "train":
            rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs, n, j, 6))
            rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs, n, j*3)
            rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, 55, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, 55*6)

            tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
            tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs, n, j*3)
            tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
            tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
            tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, 55*6)
        if use_adv and mode == 'train':
            out_d_fake = self.d_model(rec_pose)
            d_loss_adv = -torch.mean(torch.log(out_d_fake + 1e-8))
            self.tracker.update_meter("gen", mode, d_loss_adv.item())
        else:
            d_loss_adv = 0

        if self.args.train_trans:
            trans_loss = self.vel_loss(rec_trans, loaded_data["tar_trans"])
            trans_loss *= self.args.rec_weight
            self.tracker.update_meter("trans", mode, trans_loss.item())
        else:
            trans_loss = 0
        # trans_loss_vel = self.vel_loss(rec_trans[:, 1:] - rec_trans[:, :-1], loaded_data["tar_trans"][:, 1:] - loaded_data["tar_trans"][:, :-1])
        # self.tracker.update_meter("transv", mode, trans_loss_vel.item())
        # trans_loss_acc = self.vel_loss(rec_trans[:, 2:] - 2*rec_trans[:, 1:-1] + rec_trans[:, :-2], loaded_data["tar_trans"][:, 2:] - 2*loaded_data["tar_trans"][:, 1:-1] + loaded_data["tar_trans"][:, :-2])
        # self.tracker.update_meter("transa", mode, trans_loss_acc.item())

        if mode == 'train':
            return d_loss_adv + rec_loss + trans_loss # + rec_loss_vel + rec_loss_acc + trans_loss_vel + trans_loss_acc
        elif mode == 'val':
            return {
                'rec_pose': rec_pose,
                'rec_trans': rec_trans,
                'tar_pose': tar_pose,
            }
        else:
            return {
                'rec_pose': rec_pose,
                'rec_trans': rec_trans,
                'tar_pose': tar_pose,
                'tar_exps': loaded_data["tar_exps"],
                'tar_beta': loaded_data["tar_beta"],
                'tar_trans': loaded_data["tar_trans"],
            }
        
    def train(self, epoch):
        use_adv = bool(epoch>=self.args.no_adv_epoch)
        self.model.train()
        self.d_model.train()
        self.tracker.reset()
        t_start = time.time()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
    
            if use_adv:
                d_loss_final = 0
                self.opt_d.zero_grad()
                d_loss_adv = self._d_training(loaded_data)
                d_loss_final += d_loss_adv
                d_loss_final.backward()
                self.opt_d.step() 
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, use_adv, 'train')
            g_loss_final.backward()
            self.opt.step()

            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            lr_d = self.opt_d.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=lr_d)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
        self.opt_d_s.step(epoch)        
    
    
    def val(self, epoch):
        self.model.eval()
        self.d_model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.train_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_training(loaded_data, False, 'val')
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                n = tar_pose.shape[1]
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                n = tar_pose.shape[1]
                remain = n%self.args.vae_test_len
                tar_pose = tar_pose[:, :n-remain, :]
                rec_pose = rec_pose[:, :n-remain, :]
                latent_out = self.eval_copy.map2latent(rec_pose).reshape(-1, self.args.vae_length).cpu().numpy()
                latent_ori = self.eval_copy.map2latent(tar_pose).reshape(-1, self.args.vae_length).cpu().numpy()
                if its == 0:
                    latent_out_motion_all = latent_out
                    latent_ori_all = latent_ori
                else:
                    latent_out_motion_all = np.concatenate([latent_out_motion_all, latent_out], axis=0)                 
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori], axis=0)
                if self.args.debug:
                    if its == 1: break
        fid_motion = data_tools.FIDCalculator.frechet_distance(latent_out_motion_all, latent_ori_all)
        self.tracker.update_meter("fid", "val", fid_motion)
        self.val_recording(epoch)  
        
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_training(loaded_data, False, 'test')
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], 55
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    tar_beta = torch.nn.functional.interpolate(tar_beta.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    tar_exps = torch.nn.functional.interpolate(tar_exps.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1) 
                    tar_trans = torch.nn.functional.interpolate(tar_trans.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_trans = torch.nn.functional.interpolate(rec_trans.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                # print(rec_pose.shape, tar_pose.shape)
                # rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                # rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                # tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                # tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                remain = n%self.args.vae_test_len
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy()) # bs * n/8 * 240
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy())
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )
                # vertices_tar = self.smplx(
                #         betas=tar_beta.reshape(bs*n, 300), 
                #         transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                #         expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                #         jaw_pose=tar_pose[:, 66:69], 
                #         global_orient=tar_pose[:,:3], 
                #         body_pose=tar_pose[:,3:21*3+3], 
                #         left_hand_pose=tar_pose[:,25*3:40*3], 
                #         right_hand_pose=tar_pose[:,40*3:55*3], 
                #         return_joints=True, 
                #         leye_pose=tar_pose[:, 69:72], 
                #         reye_pose=tar_pose[:, 72:75],
                #     )
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                # joints_tar = vertices_tar["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    in_audio_eval, sr = librosa.load(self.args.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                    in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.args.audio_sr)
                    a_offset = int(self.align_mask * (self.args.audio_sr / self.args.pose_fps))
                    onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.args.audio_sr / self.args.pose_fps*n)], a_offset, len(in_audio_eval)-a_offset, True)
                    beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n-self.align_mask, 30, True)
                    # print(beat_vel)
                    align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n-2*self.align_mask))
               
                tar_pose_axis_np = tar_pose.detach().cpu().numpy()
                rec_pose_axis_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100) - tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100) - tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                if not self.args.train_trans:
                    tar_trans_np = tar_trans_np - tar_trans_np
                    rec_trans_np = rec_trans_np - rec_trans_np
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_axis_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_axis_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fid score: {fid}")
        self.test_recording("fid", fid, epoch) 
        
        align_avg = align/(total_length-2*len(self.test_loader)*self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)

        # data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")