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
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func


class BaseTrainer(object):
    def __init__(self, args):
        
        self.best_epoch = {
            "Rec" : 0,
            "Fid" : 0,
        } 
        self.best_metric = {
            "Rec": np.inf,
            "Fid": np.inf,
        }
        self.checkpoint_path = args.root_path+args.out_root_path+"/"+args.name
        self.batch_size = args.batch_size
        self.gpus = torch.cuda.device_count()
        # data and path
        self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/mean.npy")
        self.std_pose = np.load(args.root_path+args.std_pose_path+f"{args.pose_rep}/std.npy")
#         self.mean_audio = np.load(args.root_path+args.mean_audio_path+f"{args.audio_rep}/mean.npy")
#         self.std_audio = np.load(args.root_path+args.std_audio_path+f"{args.audio_rep}/std.npy")
#         self.mean_facial = np.load(args.root_path+args.mean_facial_path+f"{args.facial_rep}/mean.npy")
#         self.std_facial = np.load(args.root_path+args.std_facial_path+f"{args.facial_rep}/std.npy")
        
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
        
        self.no_adv_epochs = args.no_adv_epochs
        self.log_period = args.log_period
        self.test_demo = args.root_path + args.train_data_path + f"{args.pose_rep}_vis/demo.bvh"
        self.vis_lookuptable = ["000_008", "000_009", "000_010", "001_001", "002_004", "003_002"]
        
        self.loss_meters = [
            other_tools.AverageMeter('loss'), other_tools.AverageMeter('l1'), 
            other_tools.AverageMeter('gen'), other_tools.AverageMeter('dis'),] 
        self.grad_norm = 100 
        
        self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")  
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=args.batch_size,  
            shuffle=True,  
            num_workers=args.loader_workers,
            drop_last=True,
        )
        logger.info(f"Init train dataloader success")
        self.val_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "val")  
        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=args.batch_size,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
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
        self.model = getattr(model_module, args.g_name)(args)
        self.model = torch.nn.DataParallel(self.model, args.gpus).cuda()
        logger.info(self.model)
        logger.info(f"init {args.g_name} success")
        
        self.d_model = getattr(model_module, args.d_name)(args)
        self.d_model = torch.nn.DataParallel(self.d_model, args.gpus).cuda()
        logger.info(self.d_model)
        logger.info(f"init {args.d_name} success")
        
        eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
        self.eval_model = getattr(eval_model_module, args.e_name)(args)
        self.eval_model = torch.nn.DataParallel(self.eval_model, args.gpus).cuda()
        logger.info(self.eval_model)
        logger.info(f"init {args.e_name} success")
        other_tools.load_checkpoints(self.eval_model, args.root_path+args.e_path, args.e_name)
       
        self.opt = create_optimizer(args, self.model)
        self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)    
      
        
    def debug(self):
        epoch = 999 
        its = 0
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
        
        for its, (tar_pose, in_audio, in_facial, in_id) in enumerate(self.train_loader):
            tar_pose = tar_pose.reshape(-1, 123) #128*34*123 
            tar_pose = (tar_pose.numpy() * self.std_pose) + self.mean_pose
            with open(f"{results_save_path}result_raw_{self.vis_lookuptable[its]}.bvh", 'w+') as f_real:
                for line_id in range(tar_pose.shape[0]): #,args.pre_frames, args.pose_length
                    line_data = np.array2string(tar_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                    f_real.write(line_data[1:-2]+'\n')                    
            data_tools.result2target_vis(self.pose_rep, results_save_path, results_save_path, self.test_demo, False)
            break
        
    def train(self, epoch, tf_writter):
        use_adv = bool(epoch>=self.no_adv_epochs)
        self.model.train()
        self.d_model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        for its, (tar_pose, in_audio, in_facial, in_id) in enumerate(self.train_loader):
#             if its+1 == its_len and tar_pose.shape[0] < self.batchnorm_bug: # skip final bs=1, bug for bn
#                     continue
            t_data = time.time() - t_start
           
            tar_pose = tar_pose.cuda()
            in_audio = in_audio.cuda() if self.audio_rep is not "None" else None
            in_facial = in_facial.cuda() if self.facial_rep is not "None" else None
            ## in_id = in_id.cuda() if "id" in self.input_type else None
            
            in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
            in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
            in_pre_pose[:, 0:self.pre_frames, -1] = 1 
        
            t_data = time.time() - t_start 
            
            # --------------------------- d training --------------------------------- #
            d_loss_final = 0
            if use_adv:
                self.opt_d.zero_grad()
                out_pose, _, _, _ = self.model(in_pre_pose, in_audio, in_facial, in_id)
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
            out_pose, _, _, _ = self.model(in_pre_pose, in_audio, in_facial, in_id)
            huber_value = self.rec_loss(tar_pose, out_pose)
            huber_value *= self.rec_weight
            self.loss_meters[1].update(huber_value.item())
            g_loss_final += huber_value 
            if use_adv:
                dis_out = self.d_model(out_pose)
                d_fake_value = -torch.mean(torch.log(dis_out + 1e-8)) # self.adv_loss(out_d_fake, real_gt) # here 1 is real
                d_fake_value *= self.adv_weight * d_fake_value
                self.loss_meters[2].update(d_fake_value.item())
                g_loss_final += d_fake_value
                
                latent_out = self.eval_model(out_pose)
                latent_ori = self.eval_model(tar_pose)
                huber_fid_loss = self.rec_loss(latent_out, latent_ori) * self.fid_weight
                self.loss_meters[4].update(huber_fid_loss.item())
                g_loss_final += huber_fid_loss
            
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
            for its, (tar_pose, in_audio, in_facial, in_id) in enumerate(self.val_loader):
#                 if its+1 == its_len and tar_pose.shape[0] < self.batchnorm_bug: # skip final bs=1, bug for bn
#                     continue
                tar_pose = tar_pose.cuda()
                in_audio = in_audio.cuda()  
                in_facial = in_facial.cuda() if self.facial_rep is "None" else None
                # in_id = in_id.cuda() if "id" in self.input_type else None

                in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                in_pre_pose[:, 0:self.pre_frames, -1] = 1  # indicating bit for constraints
    
                out_pose, _, _, _ = self.model(in_pre_pose, in_audio, in_facial, in_id)
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
            for its, (tar_pose, in_audio, in_facial, in_id) in enumerate(self.test_loader):
                # tar_pose = tar_pose.cuda() # no mean
                n_audio = in_audio.cuda()  
                in_facial = in_facial.cuda() if self.facial_rep is "None" else None
                # in_id = in_id.cuda() if "id" in self.input_type else None
                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1
                
                in_audio = in_audio.reshape(1, -1)   
                out_dir_vec, *_ = self.model(**dict(pre_seq=pre_pose, in_audio=in_audio, in_facial=in_facial, is_test=True))
                out_final = (out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                total_length += out_final.shape[0]
                #print(out_final.shape)

                with open(f"{results_save_path}result_raw_{self.vis_lookuptable[its]}.bvh", 'w+') as f_real:
                    for line_id in range(out_final.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')                    
        data_tools.result2target_vis(self.pose_rep, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")



@logger.catch
def main_worker(args):
    logger_tools.set_args_and_logger(args)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
    tf_writter = SummaryWriter(args.root_path+args.out_root_path+args.name) 
    
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args)
    
    logger.info("Training from starch ...")          
    best_epoch_rec, best_epoch_fid = 0, 0
    best_rec, best_fid = np.inf, np.inf
    start_time = time.time()
    for epoch in range(args.epochs):
        current_rec, current_fid = trainer.val_fid(epoch, tf_writter)
        epoch_time = time.time()-start_time
        
        if current_rec < best_rec:
            best_rec = current_rec
            best_epoch_rec = epoch
            other_tools.save_checkpoints(args.root_path+args.out_root_path+"/"+args.name+f"/best_rec_{epoch}.bin", trainer.model, opt=None, epoch=None, lrs=None)
        if current_fid < best_fid:
            best_fid = current_fid
            best_epoch_fid = epoch
            other_tools.save_checkpoints(args.root_path+args.out_root_path+"/"+args.name+f"/best_fid_{epoch}.bin", trainer.model, opt=None, epoch=None, lrs=None)
        
        logger.info("RecLoss:%.2f\t"%(current_rec)+"FidLoss:%.2f\t"%(current_fid)+"Epoch Time:%.2fmin"%(epoch_time/60))
        logger.info("BestRec:%.2f\t"%(best_rec)+"BestFid:%.2f\t"%(best_fid)+"RecEpoch:%d\t"%(best_epoch_rec)+"FidEpoch:%d"%(best_epoch_fid))
        
        trainer.train(epoch, tf_writter) 
        if (epoch+1) % args.test_period == 0:
            trainer.test(epoch)
        
#         if CLOUD_TRAINING:
#             import autosearch  
#             autosearch.reporter(BestRec=best_rec)
#             autosearch.reporter(BestFid=best_fid)
      
            
if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    args = config.parse_args()
    main_worker(args)