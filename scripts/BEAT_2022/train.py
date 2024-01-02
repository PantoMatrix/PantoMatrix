# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Train script for audio2pose

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

from utils import config, logger_tools, other_tools
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
        
        # model para    
        self.pre_frames = args.pre_frames
        self.rec_loss = get_loss_func("huber_loss")
        self.adv_loss = get_loss_func("bce_loss")
        self.fid_loss = get_loss_func("l2_loss")
        self.vel_loss = get_loss_func("l2_loss")
        self.acc_loss = get_loss_func("l2_loss")
        # TODO: 
        # self.pos_loss        
        self.rec_weight = args.rec_weight
        self.adv_weight = args.adv_weight
        self.fid_weight = args.fid_weight
        self.vel_weight = args.vel_weight
        self.acc_weight = args.acc_weight
        self.grad_norm = args.grad_norm 
      
        self.no_adv_epochs = args.no_adv_epochs
        self.log_period = args.log_period
        self.test_demo = args.root_path + args.test_data_path + f"{args.pose_rep}_vis/"
        
        self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=args.batch_size,  
            shuffle=False if self.ddp else True,  
            num_workers=args.loader_workers,
            drop_last=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if self.ddp else None, 
        )
        self.train_length = len(self.train_loader)
        logger.info(f"Init train dataloader success")
       
        self.val_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "val")  
        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=args.batch_size,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
            sampler=torch.utils.data.distributed.DistributedSampler(self.val_data) if self.ddp else None, 
        )
        logger.info(f"Init val dataloader success")
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
        
        if self.ddp:
            self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=False)
        else: 
            self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        if self.rank == 0:
            logger.info(self.model)
            wandb.watch(self.model)
            logger.info(f"init {args.g_name} success")
        
        if args.d_name is not None:
            if self.ddp:
                self.d_model = getattr(model_module, args.d_name)(args).to(self.rank)
                self.d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model, process_group)   
                self.d_model = DDP(self.d_model, device_ids=[self.rank], output_device=self.rank, 
                                   broadcast_buffers=False, find_unused_parameters=False)
            else:    
                self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()
            if self.rank == 0:
                logger.info(self.d_model)
                wandb.watch(self.d_model)
                logger.info(f"init {args.d_name} success")
            self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
            self.opt_d_s = create_scheduler(args, self.opt_d)
            
        if args.e_name is not None:
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            if self.ddp:
                self.eval_model = getattr(eval_model_module, args.e_name)(args).to(self.rank)
            else:
                self.eval_model = getattr(eval_model_module, args.e_name)(args)
            if self.rank == 0:
                other_tools.load_checkpoints(self.eval_model, args.root_path+args.e_path, args.e_name)
            if self.ddp:
                self.eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.eval_model, process_group)   
                self.eval_model = DDP(self.eval_model, device_ids=[self.rank], output_device=self.rank,
                                      broadcast_buffers=False, find_unused_parameters=False)
            else:    
                self.eval_model = torch.nn.DataParallel(self.eval_model, args.gpus).cuda()    
            if self.rank == 0:
                logger.info(self.eval_model)
                wandb.watch(self.eval_model)
                logger.info(f"init {args.e_name} success")
        
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)
       
    def recording(self, epoch, its, its_len, loss_meters, lr_g, lr_d, t_data, t_train, mem_cost):
        if self.rank == 0:
            pstr = "[%03d][%03d/%03d]  "%(epoch, its, its_len)
            for name, loss_meter in self.loss_meters.items():
                if "val" not in name:
                    if loss_meter.count > 0:
                        pstr += "{}: {:.3f}\t".format(loss_meter.name, loss_meter.avg)
                        wandb.log({loss_meter.name: loss_meter.avg}, step=epoch*self.train_length+its)
                        loss_meter.reset()
            pstr += "glr: {:.1e}\t".format(lr_g)
            pstr += "dlr: {:.1e}\t".format(lr_d)
            wandb.log({'glr': lr_g, 'dlr': lr_d}, step=epoch*self.train_length+its)
            pstr += "dtime: %04d\t"%(t_data*1000)        
            pstr += "ntime: %04d\t"%(t_train*1000)
            pstr += "mem: {:.2f} ".format(mem_cost*self.gpus)
            logger.info(pstr)
     
    def val_recording(self, epoch, metrics):
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

@logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args) 
     
    logger.info("Training from starch ...")          
    start_time = time.time()
    for epoch in range(args.epochs):
        if trainer.ddp: trainer.val_loader.sampler.set_epoch(epoch)
        trainer.val(epoch)
        epoch_time = time.time()-start_time
        if trainer.rank == 0: logger.info("Time info >>>>  elapsed: %.2f mins\t"%(epoch_time/60)+"remain: %.2f mins"%((args.epochs/(epoch+1e-7)-1)*epoch_time/60))
        if trainer.ddp: trainer.train_loader.sampler.set_epoch(epoch)
        trainer.train(epoch) 
        if (epoch+1) % args.test_period == 0:
            if rank == 0:
                trainer.test(epoch)
                other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), trainer.model, opt=None, epoch=None, lrs=None)
            
    for k, v in trainer.best_epochs.items():
        wandb.log({f"{k}_best": v[0], f"{k}_epoch": v[1]})
    
    if rank == 0:
        wandb.finish()
    
            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='localhost'
    os.environ["MASTER_PORT"]='2222'
    args = config.parse_args()
    if args.ddp:
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            main_worker,
            args=(len(args.gpus), args,),
            nprocs=len(args.gpus),
                )
    else:
        main_worker(0, 1, args)