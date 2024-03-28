import os
import signal
import time
import csv
import sys
import warnings
import random
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools_hf, metric, data_transfer
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
import soundfile as sf
import librosa 

class BaseTrainer(object):
    def __init__(self, args, sp, ap, tp):
        hf_dir = "hf"
        if not os.path.exists(args.out_path + "custom/" + hf_dir + "/"):
            os.makedirs(args.out_path + "custom/" + hf_dir + "/")
        sf.write(args.out_path + "custom/" + hf_dir + "/tmp.wav", ap[1], ap[0])
        self.audio_path = args.out_path + "custom/" + hf_dir + "/tmp.wav"
        audio, ssr = librosa.load(self.audio_path)
        ap = (ssr, audio)
        self.args = args
        self.rank = 0 # dist.get_rank()
       
        #self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        self.checkpoint_path = args.out_path + "custom/" + hf_dir + "/" 
        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test", smplx_path=sp, audio_path=ap, text_path=tp)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
        logger.info(f"Init test dataloader success")
        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        
        if args.ddp:
            self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=False)
        else: 
            self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")

        self.smplx = smplx.create(
        self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()

        self.args = args
        self.joints = self.test_data.joints
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools_hf.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False,False,False])
        
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106
        self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        # print(self.vq_model_face)
        other_tools_hf.load_checkpoints(self.vq_model_face, self.args.data_path_1 + "pretrained_vq/last_790_face_v2.bin", args.e_name)
        self.args.vae_test_dim = 78
        self.vq_model_upper = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools_hf.load_checkpoints(self.vq_model_upper, self.args.data_path_1 + "pretrained_vq/upper_vertex_1layer_710.bin", args.e_name)
        self.args.vae_test_dim = 180
        self.vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools_hf.load_checkpoints(self.vq_model_hands, self.args.data_path_1 + "pretrained_vq/hands_vertex_1layer_710.bin", args.e_name)
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.vq_model_lower = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools_hf.load_checkpoints(self.vq_model_lower, self.args.data_path_1 + "pretrained_vq/lower_foot_600.bin", args.e_name)
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.global_motion = getattr(vq_model_module, "VAEConvZero")(self.args).to(self.rank)
        other_tools_hf.load_checkpoints(self.global_motion, self.args.data_path_1 + "pretrained_vq/last_1700_foot.bin", args.e_name)
        self.args.vae_test_dim = 330
        self.args.vae_layer = 4
        self.args.vae_length = 240

        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()
        self.global_motion.eval()

        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)
      
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank) 
        in_word = None# dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)

        # tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        # tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        tar4dis = torch.cat([tar_pose_jaw, tar_pose_upper, tar_pose_hands, tar_pose_leg], dim=2)

        tar_index_value_face_top = self.vq_model_face.map2index(tar_pose_face) # bs*n/4
        tar_index_value_upper_top = self.vq_model_upper.map2index(tar_pose_upper) # bs*n/4
        tar_index_value_hands_top = self.vq_model_hands.map2index(tar_pose_hands) # bs*n/4
        tar_index_value_lower_top = self.vq_model_lower.map2index(tar_pose_lower) # bs*n/4
      
        latent_face_top = self.vq_model_face.map2latent(tar_pose_face) # bs*n/4
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)
        
        index_in = torch.stack([tar_index_value_upper_top, tar_index_value_hands_top, tar_index_value_lower_top], dim=-1).long()
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        # print(tar_index_value_upper_top.shape, index_in.shape)
        return {
            "tar_pose_jaw": tar_pose_jaw,
            "tar_pose_face": tar_pose_face,
            "tar_pose_upper": tar_pose_upper,
            "tar_pose_lower": tar_pose_lower,
            "tar_pose_hands": tar_pose_hands,
            'tar_pose_leg': tar_pose_leg,
            "in_audio": in_audio,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "tar4dis": tar4dis,
            "tar_index_value_face_top": tar_index_value_face_top,
            "tar_index_value_upper_top": tar_index_value_upper_top,
            "tar_index_value_hands_top": tar_index_value_hands_top,
            "tar_index_value_lower_top": tar_index_value_lower_top,
            "latent_face_top": latent_face_top,
            "latent_upper_top": latent_upper_top,
            "latent_hands_top": latent_hands_top,
            "latent_lower_top": latent_lower_top,
            "latent_in":  latent_in,
            "index_in": index_in,
            "tar_id": tar_id,
            "latent_all": latent_all,
            "tar_pose_6d": tar_pose_6d,
            "tar_contact": tar_contact,
        }
    
    def _g_test(self, loaded_data):
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        in_word =None# loaded_data["in_word"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        in_audio = loaded_data["in_audio"]
        tar_trans = loaded_data["tar_trans"]

        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            # in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        
        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames

        for i in range(0, roundt):
            # in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
            else:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
                # print(latent_all_tmp.shape, latent_last.shape)
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            net_out_val = self.model(
                in_audio = in_audio_tmp,
                in_word=None, #in_word_tmp,
                mask=mask_val,
                in_motion = latent_all_tmp,
                in_id = in_id_tmp,
                use_attentions=True,)
            
            if self.args.cu != 0:
                rec_index_upper = self.log_softmax(net_out_val["cls_upper"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_upper = torch.max(rec_index_upper.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_upper = self.vq_model_upper.decode(rec_index_upper)
            else:
                _, rec_index_upper, _, _ = self.vq_model_upper.quantizer(net_out_val["rec_upper"])
                #rec_upper = self.vq_model_upper.decoder(rec_index_upper)
            if self.args.cl != 0:
                rec_index_lower = self.log_softmax(net_out_val["cls_lower"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_lower = torch.max(rec_index_lower.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_lower = self.vq_model_lower.decode(rec_index_lower)
            else:
                _, rec_index_lower, _, _ = self.vq_model_lower.quantizer(net_out_val["rec_lower"])
                #rec_lower = self.vq_model_lower.decoder(rec_index_lower)
            if self.args.ch != 0:
                rec_index_hands = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_hands = torch.max(rec_index_hands.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_hands = self.vq_model_hands.decode(rec_index_hands)
            else:
                _, rec_index_hands, _, _ = self.vq_model_hands.quantizer(net_out_val["rec_hands"])
                #rec_hands = self.vq_model_hands.decoder(rec_index_hands)
            if self.args.cf != 0:
                rec_index_face = self.log_softmax(net_out_val["cls_face"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_face = torch.max(rec_index_face.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_face = self.vq_model_face.decoder(rec_index_face)
            else:
                _, rec_index_face, _, _ = self.vq_model_face.quantizer(net_out_val["rec_face"])
                #rec_face = self.vq_model_face.decoder(rec_index_face)

            if i == 0:
                rec_index_all_face.append(rec_index_face)
                rec_index_all_upper.append(rec_index_upper)
                rec_index_all_lower.append(rec_index_lower)
                rec_index_all_hands.append(rec_index_hands)
            else:
                rec_index_all_face.append(rec_index_face[:, self.args.pre_frames:])
                rec_index_all_upper.append(rec_index_upper[:, self.args.pre_frames:])
                rec_index_all_lower.append(rec_index_lower[:, self.args.pre_frames:])
                rec_index_all_hands.append(rec_index_hands[:, self.args.pre_frames:])

            if self.args.cu != 0:
                rec_upper_last = self.vq_model_upper.decode(rec_index_upper)
            else:
                rec_upper_last = self.vq_model_upper.decoder(rec_index_upper)
            if self.args.cl != 0:
                rec_lower_last = self.vq_model_lower.decode(rec_index_lower)
            else:
                rec_lower_last = self.vq_model_lower.decoder(rec_index_lower)
            if self.args.ch != 0:
                rec_hands_last = self.vq_model_hands.decode(rec_index_hands)
            else:
                rec_hands_last = self.vq_model_hands.decoder(rec_index_hands)
            # if self.args.cf != 0:
            #     rec_face_last = self.vq_model_face.decode(rec_index_face)
            # else:
            #     rec_face_last = self.vq_model_face.decoder(rec_index_face)

            rec_pose_legs = rec_lower_last[:, :, :54]
            bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            rec_pose_upper = rec_upper_last.reshape(bs, n, 13, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
            rec_pose_hands = rec_hands_last.reshape(bs, n, 30, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            rec_x_trans = other_tools_hf.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            rec_z_trans = other_tools_hf.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:,:,1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            latent_last = torch.cat([rec_pose, rec_trans, rec_lower_last[:, :, 57:61]], dim=-1)

        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)
        if self.args.cu != 0:
            rec_upper = self.vq_model_upper.decode(rec_index_upper)
        else:
            rec_upper = self.vq_model_upper.decoder(rec_index_upper)
        if self.args.cl != 0:
            rec_lower = self.vq_model_lower.decode(rec_index_lower)
        else:
            rec_lower = self.vq_model_lower.decoder(rec_index_lower)
        if self.args.ch != 0:
            rec_hands = self.vq_model_hands.decode(rec_index_hands)
        else:
            rec_hands = self.vq_model_hands.decoder(rec_index_hands)
        if self.args.cf != 0:
            rec_face = self.vq_model_face.decode(rec_index_face)
        else:
            rec_face = self.vq_model_face.decoder(rec_index_face)

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
        rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools_hf.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools_hf.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }


    def test_demo(self, epoch):
        '''
        input audio and text, output motion
        do not calculate loss and metric
        save video
        '''
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            import shutil
            shutil.rmtree(results_save_path)
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        # self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                # interpolate to 30fps  
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                # print(rec_pose.shape, tar_pose.shape)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)

                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                        
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                #'''
                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                #'''
                total_length += n
                render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                    results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz', 
                    # results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz', 
                    results_save_path,
                    self.audio_path,
                    self.args.data_path_1+"smplx_models/",
                    use_matplotlib = False,
                    args = self.args,
                    )
                # render_vid_path = other_tools_hf.render_one_sequence(
                #     results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz', 
                #     results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz', 
                #     results_save_path,
                #     self.audio_path,
                #     self.args.data_path_1+"smplx_models/",
                #     use_matplotlib = False,
                #     args = self.args,
                #     )
        result = gr.Video(value=render_vid_path, visible=True)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        return result
       
@logger.catch
def emage(audio_path):
    smplx_path = None
    text_path = None
    rank = 0
    world_size = 1
    args = config.parse_args()
    #os.environ['TRANSFORMERS_CACHE'] = args.data_path_1 + "hub/"
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    #logger_tools.set_args_and_logger(args, rank)
    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    # return one intance of trainer
    trainer = BaseTrainer(args, sp = smplx_path, ap = audio_path, tp = text_path)
    other_tools_hf.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    result = trainer.test_demo(999)
    return result

examples = [
    ["scripts/EMAGE_2024/audio_example/2_scott_0_1_1.wav"],
    ["scripts/EMAGE_2024/audio_example/2_scott_0_2_2.wav"],
    ["scripts/EMAGE_2024/audio_example/2_scott_0_3_3.wav"]
]

demo = gr.Interface(
    emage,  # function
    inputs=[
        # gr.File(label="Please upload SMPL-X file with npz format here.", file_types=["npz", "NPZ"]),
        gr.Audio(),
        # gr.File(label="Please upload textgrid format file here.", file_types=["TextGrid", "Textgrid", "textgrid"])
    ],  # input type
    outputs=gr.Video(format="mp4", visible=True),
    title='EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling',
    description="1. Upload your audio (less than 60s).  <br/>\
        2. Then, sit back and wait for the rendering to happen! This may take a while (e.g. 3 minutes) <br/>\
        3. After, you can view the videos.  <br/>",
    article="Relevant links: [Project Page](https://pantomatrix.github.io/EMAGE/)", 
    examples=examples,
)

            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.1'
    os.environ["MASTER_PORT"]='8675'
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    demo.launch(share=True)
