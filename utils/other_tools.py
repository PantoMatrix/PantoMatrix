import os
import numpy as np
import random
import torch
import csv
import pprint
from loguru import logger
from collections import OrderedDict

def print_exp_info(args):
    logger.info(pprint.pformat(vars(args)))
    logger.info(f"# ------------ {args.name} ----------- #")
    logger.info("PyTorch version: {}".format(torch.__version__))
    logger.info("CUDA version: {}".format(torch.version.cuda))
    logger.info("{} GPUs".format(torch.cuda.device_count()))
    logger.info(f"Random Seed: {args.random_seed}")

def args2csv(args, get_head=False, list4print=[]):
    for k, v in args.items():
        if isinstance(args[k], dict):
            args2csv(args[k], get_head, list4print)
        else: list4print.append(k) if get_head else list4print.append(v)
    return list4print

def record_trial(args, csv_path, best_metric, best_epoch):
    metric_name = []
    metric_value = []
    metric_epoch = []
    list4print = []
    name4print = []
    for k, v in vars(args).items():
        list4print.append(v)
        name4print.append(k)
    
    for k, v in best_metric.items():
        metric_name.append(k)
        metric_value.append(v)
        metric_epoch.append(best_epoch[k])
    
    if not os.path.exists(csv_path):
        with open(csv_path, "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([*metric_name, *metric_name, *name4print])
            
    with open(csv_path, "a+") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([*metric_value,*metric_epoch, *list4print])
        

def set_random_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.deterministic #args.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.enabled = args.cudnn_enabled
    

def save_checkpoints(save_path, model, opt=None, epoch=None, lrs=None):
    if lrs is not None:
        states = { 'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'opt_state': opt.state_dict(),
                'lrs':lrs.state_dict(),}
    elif opt is not None:
        states = { 'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'opt_state': opt.state_dict(),}
    else:
        states = { 'model_state': model.state_dict(),}
    torch.save(states, save_path)


def load_checkpoints(model, save_path, load_name='model'):
    states = torch.load(save_path)
    new_weights = OrderedDict()
    flag=False
    for k, v in states['model_state'].items():
        if "module" not in k:
            break
        else:
            new_weights[k[7:]]=v
            flag=True
    if flag: 
        model.load_state_dict(new_weights)
    else:
        model.load_state_dict(states['model_state'])
    logger.info(f"load self-pretrained checkpoints for {load_name}")


def model_complexity(model, args):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model,  (args.T_GLOBAL._DIM, args.TRAIN.CROP, args.TRAIN), 
        as_strings=False, print_per_layer_stat=False)
    logging.info('{:<30}  {:<8} BFlops'.format('Computational complexity: ', flops / 1e9))
    logging.info('{:<30}  {:<8} MParams'.format('Number of parameters: ', params / 1e6))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)