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
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank()
        self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        if self.rank==0:
            if self.args.stat == "ts":
                self.writer = SummaryWriter(log_dir=args.out_path + "custom/" + args.name + args.notes + "/")
            else:
                wandb.init(project=args.project, entity="liu1997", dir=args.out_path, name=args.name[12:] + args.notes)
                wandb.config.update(args)
                self.writer = None  
        #self.test_demo = args.data_path + args.test_data_path + "bvh_full/"        
        # self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
        # self.train_loader = torch.utils.data.DataLoader(
        #     self.train_data, 
        #     batch_size=args.batch_size,  
        #     shuffle=False if args.ddp else True,  
        #     num_workers=args.loader_workers,
        #     drop_last=True,
        #     sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if args.ddp else None, 
        # )
        # self.train_length = len(self.train_loader)
        # logger.info(f"Init train dataloader success")
       
        # self.val_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "val")  
        # self.val_loader = torch.utils.data.DataLoader(
        #     self.val_data, 
        #     batch_size=args.batch_size,  
        #     shuffle=False,  
        #     num_workers=args.loader_workers,
        #     drop_last=False,
        #     sampler=torch.utils.data.distributed.DistributedSampler(self.val_data) if args.ddp else None, 
        # )
        # logger.info(f"Init val dataloader success")
        if self.rank == 0:
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
            if args.stat == "wandb":
                wandb.watch(self.model)
        
        # if args.d_name is not None:
        #     if args.ddp:
        #         self.d_model = getattr(model_module, args.d_name)(args).to(self.rank)
        #         self.d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model, process_group)   
        #         self.d_model = DDP(self.d_model, device_ids=[self.rank], output_device=self.rank, 
        #                            broadcast_buffers=False, find_unused_parameters=False)
        #     else:    
        #         self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()
        #     if self.rank == 0:
        #         logger.info(self.d_model)
        #         logger.info(f"init {args.d_name} success")
        #         if args.stat == "wandb":
        #             wandb.watch(self.d_model)
        #     self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
        #     self.opt_d_s = create_scheduler(args, self.opt_d)
           
        if args.e_name is not None:
            """
            bugs on DDP training using eval_model, using additional eval_copy for evaluation 
            """
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            # eval copy is for single card evaluation
            if self.args.ddp:
                self.eval_model = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank) 
            else:
                self.eval_model = getattr(eval_model_module, args.e_name)(args)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                
            #if self.rank == 0:
            other_tools.load_checkpoints(self.eval_copy, args.data_path+args.e_path, args.e_name)
            other_tools.load_checkpoints(self.eval_model, args.data_path+args.e_path, args.e_name)
            if self.args.ddp:
                self.eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.eval_model, process_group)   
                self.eval_model = DDP(self.eval_model, device_ids=[self.rank], output_device=self.rank,
                                      broadcast_buffers=False, find_unused_parameters=False)
            self.eval_model.eval()
            self.eval_copy.eval()
            if self.rank == 0:
                logger.info(self.eval_model)
                logger.info(f"init {args.e_name} success")  
                if args.stat == "wandb":
                    wandb.watch(self.eval_model) 
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
        self.alignmenter = metric.alignment(0.3, 7, self.train_data.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]) if self.rank == 0 else None
        self.align_mask = 60
        self.l1_calculator = metric.L1div() if self.rank == 0 else None
       
    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = "[%03d][%03d/%03d]  "%(epoch, its, self.train_length)
        for name, states in self.tracker.loss_meters.items():
            metric = states['train']
            if metric.count > 0:
                pstr += "{}: {:.3f}\t".format(name, metric.avg)
                self.writer.add_scalar(f"train/{name}", metric.avg, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({name: metric.avg}, step=epoch*self.train_length+its)
        pstr += "glr: {:.1e}\t".format(lr_g)
        self.writer.add_scalar("lr/glr", lr_g, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'glr': lr_g}, step=epoch*self.train_length+its)
        if lr_d is not None:
            pstr += "dlr: {:.1e}\t".format(lr_d)
            self.writer.add_scalar("lr/dlr", lr_d, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'dlr': lr_d}, step=epoch*self.train_length+its)
        pstr += "dtime: %04d\t"%(t_data*1000)        
        pstr += "ntime: %04d\t"%(t_train*1000)
        pstr += "mem: {:.2f} ".format(mem_cost*len(self.args.gpus))
        logger.info(pstr)
     
    def val_recording(self, epoch):
        pstr_curr = "Curr info >>>>  "
        pstr_best = "Best info >>>>  "
        for name, states in self.tracker.loss_meters.items():
            metric = states['val']
            if metric.count > 0:
                pstr_curr += "{}: {:.3f}     \t".format(name, metric.avg)
                if epoch != 0:
                    if self.args.stat == "ts":
                        self.writer.add_scalars(f"val/{name}", {name+"_val":metric.avg, name+"_train":states['train'].avg}, epoch*self.train_length)
                    else:
                        wandb.log({name+"_val": metric.avg, name+"_train":states['train'].avg}, step=epoch*self.train_length)
                    new_best_train, new_best_val = self.tracker.update_and_plot(name, epoch, self.checkpoint_path+f"{name}_{self.args.name+self.args.notes}.png")
                    if new_best_val:
                        other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"{name}.bin"), self.model, opt=None, epoch=None, lrs=None)        
        for k, v in self.tracker.values.items():
            metric = v['val']['best']
            if self.tracker.loss_meters[k]['val'].count > 0:
                pstr_best += "{}: {:.3f}({:03d})\t".format(k, metric['value'], metric['epoch'])
        logger.info(pstr_curr)
        logger.info(pstr_best)
   
    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)

@logger.catch
def main_worker(rank, world_size, args):
    #os.environ['TRANSFORMERS_CACHE'] = args.data_path_1 + "hub/"
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args) 
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    trainer.test(999)
    
    
            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.1'
    os.environ["MASTER_PORT"]='8675'
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
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