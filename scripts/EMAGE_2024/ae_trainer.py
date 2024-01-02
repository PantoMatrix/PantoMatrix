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
        self.joints = self.train_data.joints
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        self.tracker = other_tools.EpochTracker(["rec", "vel", "ver", "com", "kl", "acc"], [False, False, False, False, False, False])
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")
        self.rec_loss = get_loss_func("GeodesicLoss")
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
    
    def inverse_selection(self, filtered_t, selection_array, n):
        # 创建一个全为零的数组，形状为 n*165
        original_shape_t = np.zeros((n, selection_array.size))
        
        # 找到选择数组中为1的索引位置
        selected_indices = np.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # 创建一个全为零的数组，形状为 n*165
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        
        # 找到选择数组中为1的索引位置
        selected_indices = torch.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t

    def train(self, epoch):
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, dict_data in enumerate(self.train_loader):
            tar_pose = dict_data["pose"]
            tar_beta = dict_data["beta"].cuda()
            tar_trans = dict_data["trans"].cuda()
            tar_pose = tar_pose.cuda()  
            bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
            tar_exps = torch.zeros((bs, n, 100)).cuda()
            tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
            tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
            t_data = time.time() - t_start 
            
            self.opt.zero_grad()
            g_loss_final = 0
            net_out = self.model(tar_pose)
            rec_pose = net_out["rec_pose"]
            rec_pose = rec_pose.reshape(bs, n, j, 6)
            rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
            tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
            loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec", "train", loss_rec.item())
            g_loss_final += loss_rec

            velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) * self.args.rec_weight
            acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) * self.args.rec_weight
            self.tracker.update_meter("vel", "train", velocity_loss.item())
            self.tracker.update_meter("acc", "train", acceleration_loss.item())
            g_loss_final += velocity_loss 
            g_loss_final += acceleration_loss 
             # vertices loss
            if self.args.rec_ver_weight > 0:
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
                tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
                vertices_rec = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=rec_pose[:, 66:69], 
                    global_orient=rec_pose[:,:3], 
                    body_pose=rec_pose[:,3:21*3+3], 
                    left_hand_pose=rec_pose[:,25*3:40*3], 
                    right_hand_pose=rec_pose[:,40*3:55*3], 
                    return_verts=True,
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )
                vertices_tar = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3], 
                    body_pose=tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3], 
                    right_hand_pose=tar_pose[:,40*3:55*3], 
                    return_verts=True,
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )  
                vectices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])
                self.tracker.update_meter("ver", "train", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                g_loss_final += vectices_loss*self.args.rec_weight*self.args.rec_ver_weight

                vertices_vel_loss = self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1]) * self.args.rec_weight
                vertices_acc_loss = self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1]) * self.args.rec_weight
                g_loss_final += vertices_vel_loss * self.args.rec_weight * self.args.rec_ver_weight
                g_loss_final += vertices_acc_loss * self.args.rec_weight * self.args.rec_ver_weight 
            
            # if self.args.vel_weight > 0:  
            #     pos_rec_vel = other_tools.estimate_linear_velocity(vertices_rec['joints'], 1/self.pose_fps)
            #     pos_tar_vel = other_tools.estimate_linear_velocity(vertices_tar['joints'], 1/self.pose_fps)
            #     vel_rec_loss = self.vel_loss(pos_rec_vel, pos_tar_vel)
            #     tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
            #     rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            #     rot_rec_vel = other_tools.estimate_angular_velocity(rec_pose, 1/self.pose_fps)
            #     rot_tar_vel = other_tools.estimate_angular_velocity(tar_pose, 1/self.pose_fps)
            #     vel_rec_loss += self.vel_loss(pos_rec_vel, pos_tar_vel)
            #     self.tracker.update_meter("vel", "train", vel_rec_loss.item()*self.args.vel_weight)
            #     loss += (vel_rec_loss*self.args.vel_weight)

            # ---------------------- vae -------------------------- #
            if "VQVAE" in self.args.g_name:
                loss_embedding = net_out["embedding_loss"]
                g_loss_final += loss_embedding
                self.tracker.update_meter("com", "train", loss_embedding.item())
            # elif "VAE" in self.args.g_name:
            #     pose_mu, pose_logvar = net_out["pose_mu"], net_out["pose_logvar"] 
            #     KLD = -0.5 * torch.sum(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
            #     if epoch < 0:
            #         KLD_weight = 0
            #     else:
            #         KLD_weight = min(1.0, (epoch - 0) * 0.05) * 0.01
            #     loss += KLD_weight * KLD
            #     self.tracker.update_meter("kl", "train", KLD_weight * KLD.item())    
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        t_start = time.time()
        with torch.no_grad():
            for its, dict_data in enumerate(self.val_loader):
                tar_pose = dict_data["pose"]
                tar_beta = dict_data["beta"].cuda()
                tar_trans = dict_data["trans"].cuda()
                tar_pose = tar_pose.cuda()  
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                tar_exps = torch.zeros((bs, n, 100)).cuda()
                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                t_data = time.time() - t_start 

                #self.opt.zero_grad()
                #g_loss_final = 0
                net_out = self.model(tar_pose)
                rec_pose = net_out["rec_pose"]
                rec_pose = rec_pose.reshape(bs, n, j, 6)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
                self.tracker.update_meter("rec", "val", loss_rec.item())
                #g_loss_final += loss_rec

                 # vertices loss
                if self.args.rec_ver_weight > 0:
                    tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                    rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                    rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
                    tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_verts=True, 
                        leye_pose=tar_pose[:, 69:72], 
                        reye_pose=tar_pose[:, 72:75],
                    )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
                        jaw_pose=tar_pose[:, 66:69], 
                        global_orient=tar_pose[:,:3], 
                        body_pose=tar_pose[:,3:21*3+3], 
                        left_hand_pose=tar_pose[:,25*3:40*3], 
                        right_hand_pose=tar_pose[:,40*3:55*3], 
                        return_verts=True, 
                        leye_pose=tar_pose[:, 69:72], 
                        reye_pose=tar_pose[:, 72:75],
                    )  
                    vectices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])
                    self.tracker.update_meter("ver", "val", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                if "VQVAE" in self.args.g_name:
                    loss_embedding = net_out["embedding_loss"]
                    self.tracker.update_meter("com", "val", loss_embedding.item())
                    #g_loss_final += vectices_loss*self.args.rec_weight*self.args.rec_ver_weight
            self.val_recording(epoch)
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        self.model.eval()
        with torch.no_grad():
            for its, dict_data in enumerate(self.test_loader):
                tar_pose = dict_data["pose"]
                tar_pose = tar_pose.cuda()
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                remain = n%self.args.pose_length
                tar_pose = tar_pose[:, :n-remain, :]
                #print(tar_pose.shape)
                if True:
                    net_out = self.model(tar_pose)
                    rec_pose = net_out["rec_pose"]
                    n = rec_pose.shape[1]
                    tar_pose = tar_pose[:, :n, :]
                    rec_pose = rec_pose.reshape(bs, n, j, 6) 
                    rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                    rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                    rec_pose = rec_pose.cpu().numpy()
                else:
                    pass
#                     for i in range(tar_pose.shape[1]//(self.args.vae_test_len)):
#                         tar_pose_new = tar_pose[:,i*(self.args.vae_test_len):i*(self.args.vae_test_len)+self.args.vae_test_len,:]
#                         net_out = self.model(**dict(inputs=tar_pose_new))
#                         rec_pose = net_out["rec_pose"]
#                         rec_pose = (rec_pose.reshape(rec_pose.shape[0], rec_pose.shape[1], -1, 6) * self.joint_level_mask_cuda).reshape(rec_pose.shape[0], rec_pose.shape[1], -1)
#                         if "rot6d" in self.args.pose_rep:
#                             rec_pose = data_transfer.rotation_6d_to_matrix(rec_pose.reshape(tar_pose.shape[0], self.args.vae_test_len, -1, 6))
#                             rec_pose = data_transfer.matrix_to_euler_angles(rec_pose, "XYZ").reshape(rec_pose.shape[0], rec_pose.shape[1], -1)
#                             if "smplx" not in self.args.pose_rep:
#                                 rec_pose = torch.rad2deg(rec_pose)
#                             rec_pose = rec_pose * self.joint_mask_cuda
                            
#                         out_sub = rec_pose.cpu().numpy().reshape(-1, rec_pose.shape[2])
#                         if i != 0:
#                             out_final = np.concatenate((out_final,out_sub), 0)
#                         else:
#                             out_final = out_sub
                            
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                tar_pose = tar_pose.cpu().numpy()
                
                total_length += n 
                # --- save --- #
                if 'smplx' in self.args.pose_rep:
                    gt_npz = np.load(self.args.data_path+self.args.pose_rep+"/"+test_seq_list.iloc[its]['id']+'.npz', allow_pickle=True)
                    stride = int(30 / self.args.pose_fps)
                    tar_pose = self.inverse_selection(tar_pose, self.test_data.joint_mask, tar_pose.shape[0])
                    np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                        betas=gt_npz["betas"],
                        poses=tar_pose[:n],
                        expressions=gt_npz["expressions"]-gt_npz["expressions"],
                        trans=gt_npz["trans"][::stride][:n] - gt_npz["trans"][::stride][:n],
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate = 30 ,
                    )
                    rec_pose = self.inverse_selection(rec_pose, self.test_data.joint_mask, rec_pose.shape[0])
                    np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                        betas=gt_npz["betas"],
                        poses=rec_pose,
                        expressions=gt_npz["expressions"]-gt_npz["expressions"],
                        trans=gt_npz["trans"][::stride][:n] - gt_npz["trans"][::stride][:n],
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate = 30 ,
                    )       
                else:
                    rec_pose = rc.axis_angle_to_matrix(torch.from_numpy(rec_pose.reshape(bs*n, j, 3)))
                    rec_pose = np.rad2deg(rc.matrix_to_euler_angles(rec_pose, "XYZ")).reshape(bs*n, j*3).numpy()                
                    tar_pose = rc.axis_angle_to_matrix(torch.from_numpy(tar_pose.reshape(bs*n, j, 3)))
                    tar_pose = np.rad2deg(rc.matrix_to_euler_angles(tar_pose, "XYZ")).reshape(bs*n, j*3).numpy() 
                    #trans="0.000000 0.000000 0.000000"
                    
                    with open(f"{self.args.data_path}{self.args.pose_rep}/{test_seq_list.iloc[its]['id']}.bvh", "r") as f_demo:
                        with open(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.bvh', 'w+') as f_gt:
                            with open(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.bvh', 'w+') as f_real:
                                for i, line_data in enumerate(f_demo.readlines()):
                                    if i < 431:
                                        f_real.write(line_data)
                                        f_gt.write(line_data)
                                    else: break
                                for line_id in range(n): #,args.pre_frames, args.pose_length
                                    line_data = np.array2string(rec_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                                    f_real.write(line_data[1:-2]+'\n')
                                for line_id in range(n): #,args.pre_frames, args.pose_length
                                    line_data = np.array2string(tar_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                                    f_gt.write(line_data[1:-2]+'\n')
                # with open(results_save_path+"gt_"+test_seq_list[its]+'.pkl', 'wb') as fw:
                #     pickle.dump(new_dict, fw)
                # #new_dict2["fullpose"] = out_final
                # with open(results_save_path+"res_"+test_seq_list[its]+'.pkl', 'wb') as fw1:
                #     pickle.dump(new_dict2, fw1)

                # other_tools.render_one_sequence(
                #     results_save_path+"res_"+test_seq_list[its]+'.pkl',
                #     results_save_path+"gt_"+test_seq_list[its]+'.pkl',
                #     results_save_path,
                #     self.args.data_path + self.args.test_data_path + 'wave16k/' + test_seq_list[its]+'.npy',
                # )
                                                                                                
                #if its == 1:break
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")