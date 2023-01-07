# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Test script for audio2pose

import os
import signal
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
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import wandb

from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func

class BaseTrainer(object):
    def __init__(self, args):
        self.notes = args.notes
        self.ddp = args.ddp
        self.rank = dist.get_rank()
        self.checkpoint_path = args.root_path+args.out_root_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.root_path+args.out_root_path+"/"+args.name
        self.batch_size = args.batch_size
        self.gpus = len(args.gpus)
        self.trainer_name = args.trainer
        self.best_epochs = {
            'fid_val': [np.inf, 0],
            'rec_val': [np.inf, 0],
                           }
        self.loss_meters = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'), 
        } 
        self.pose_version = args.pose_version
        # data and path
        self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
        
        # pose
        self.pose_rep = args.pose_rep 
        self.pose_fps = args.pose_fps
        self.pose_dims = args.pose_dims
        # audio
        self.audio_rep = args.audio_rep
        self.audio_fps = args.audio_fps
        #self.audio_dims = args.audio_dims
        # facial
        self.facial_rep = args.facial_rep
        self.facial_fps = args.facial_fps
        self.facial_dims = args.facial_dims
        self.pose_length = args.pose_length
        self.stride = args.stride
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.speaker_id = args.speaker_id
        self.alignmenter = metric.alignment(0.3, 2)
        self.srgr_calculator = metric.SRGR(4, 47)
        self.l1_calculator = metric.L1div()
        # model para    
        self.pre_frames = args.pre_frames
        self.rec_loss = get_loss_func("huber_loss")
        self.adv_loss = get_loss_func("bce_loss")
        self.fid_loss = get_loss_func("l2_loss")
        self.vel_loss = get_loss_func("l2_loss")
        self.acc_loss = get_loss_func("l2_loss")      
        self.rec_weight = args.rec_weight
        self.adv_weight = args.adv_weight
        self.fid_weight = args.fid_weight
        self.vel_weight = args.vel_weight
        self.acc_weight = args.acc_weight
        self.grad_norm = args.grad_norm 
      
        self.no_adv_epochs = args.no_adv_epochs
        self.log_period = args.log_period
        self.test_demo = args.root_path + args.test_data_path + f"{args.pose_rep}_vis/"
       
        self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=1,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
        )
        logger.info(f"Init test dataloader success")
        
        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        self.model = getattr(model_module, args.g_name)(args)
        other_tools.load_checkpoints(self.model, args.root_path+args.test_ckpt, args.g_name)
        self.model = torch.nn.DataParallel(self.model, args.gpus).cuda()
        if self.rank == 0:
            logger.info(self.model)
            wandb.watch(self.model)
            logger.info(f"init {args.g_name} success")
            
        if args.e_name is not None:
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            self.eval_model = getattr(eval_model_module, args.e_name)(args)
            if self.rank == 0:
                other_tools.load_checkpoints(self.eval_model, args.root_path+args.e_path, args.e_name)   
            self.eval_model = torch.nn.DataParallel(self.eval_model, args.gpus).cuda()    
            if self.rank == 0:
                logger.info(self.eval_model)
                wandb.watch(self.eval_model)
                logger.info(f"init {args.e_name} success")
             
    def test_recording(self, epoch, metrics):
        if self.rank == 0: 
            pstr_curr = "Curr info >>>>  "
            pstr_best = "Best info >>>>  "

            for name, metric in metrics.items():
                if "val" in name:
                    if metric.count > 0:
                        pstr_curr += "{}: {:.3f}     \t".format(metric.name, metric.avg)
                        wandb.log({metric.name: metric.avg}, step=epoch*self.train_length)
                        if metric.avg < self.best_epochs[metric.name][0]:
                            self.best_epochs[metric.name][0] = metric.avg
                            self.best_epochs[metric.name][1] = epoch
                            other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"{metric.name}.bin"), self.model, opt=None, epoch=None, lrs=None)        
                        metric.reset()
            for k, v in self.best_epochs.items():
                pstr_best += "{}: {:.3f}({:03d})\t".format(k, v[0], v[1])
            logger.info(pstr_curr)
            logger.info(pstr_best)
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        t_start = 10
        t_end = 500
        align = 0 
        self.model.eval()
        with torch.no_grad():
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            for its, batch_data in enumerate(self.test_loader):
                tar_pose = batch_data["pose"].cuda()
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
                in_sem = batch_data["sem"].cuda() if self.sem_rep is not None else None
                
                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1
                
                in_audio = in_audio.reshape(1, -1)
                if self.trainer_name == "multi":
                    out_dir_vec, *_ = self.model(**dict(pre_seq=pre_pose, in_audio=in_audio, in_word=in_word, in_id=in_id))
                else:
                    out_dir_vec = self.model(**dict(pre_seq=pre_pose, in_audio=in_audio, in_text=in_word, in_facial=in_facial, in_id=in_id, in_emo=in_emo))
                num_divs = (tar_pose.shape[1]-self.pose_length)//self.stride+1
                
                for i in range(num_divs):
                    if i == 0:
                        cat_results = out_dir_vec[:,i*self.stride:i*self.stride+self.pose_length, :]
                        cat_targets = tar_pose[:,i*self.stride:i*self.stride+self.pose_length, :]
                        #cat_sem = in_sem[:,i*self.stride:i*self.stride+self.pose_length]
                    else:
                        cat_results = torch.cat([cat_results, out_dir_vec[:,i*self.stride:i*self.stride+self.pose_length, :]], 0)
                        cat_targets = torch.cat([cat_targets, tar_pose[:,i*self.stride:i*self.stride+self.pose_length, :]], 0)
                        #cat_sem = torch.cat([cat_sem, in_sem[:,i*self.stride:i*self.stride+self.pose_length]], 0)
                np_cat_results = (cat_results.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                #np_cat_targets = (cat_targets.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                _ = self.l1_calculator.run(np_cat_results)
                latent_out = self.eval_model(cat_results)
                latent_ori = self.eval_model(cat_targets)
                if its == 0:
                    latent_out_all = latent_out.cpu().numpy()
                    latent_ori_all = latent_ori.cpu().numpy()
                else:
                    latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()], axis=0)
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()], axis=0)
                    
                out_final = (out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                np_cat_results = out_final
                np_cat_targets = (tar_pose.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                _ = self.srgr_calculator.run(np_cat_results, np_cat_targets, in_sem.cpu().numpy())
                total_length += out_final.shape[0]
                
                onset_raw, onset_bt, onset_bt_rms = self.alignmenter.load_audio(in_audio.cpu().numpy().reshape(-1), t_start, t_end, True)
                beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.alignmenter.load_pose(out_final, t_start, t_end, self.pose_fps, True)
                align += self.alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, self.pose_fps)
             
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(out_final.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')  
        align_avg = align/len(self.test_loader)
        logger.info(f"align score: {align_avg}")
        srgr = self.srgr_calculator.avg()
        logger.info(f"srgr score: {srgr}")
        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fid score: {fid}")
        data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")

@logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args) 
    
    trainer = BaseTrainer(args) 
    logger.info("Testing from ckpt ...")
    epoch = 9999
    trainer.test(epoch)
              
            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='localhost'
    os.environ["MASTER_PORT"]='2222'
    args = config.parse_args()
    main_worker(0, 1, args)