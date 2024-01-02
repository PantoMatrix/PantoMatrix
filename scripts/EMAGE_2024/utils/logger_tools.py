import os
import inspect
import sys
import yaml
#import wandb
from loguru import logger

def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<blue>{time: MM-DD HH:mm:ss}</blue> | "
        #"<level>{level: <8}</level> | "
        #"<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        "<level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file,
            format=loguru_format,       
                  )


def set_args_and_logger(args, rank):
    """
    set logger file and print args
    """
    args_name_dir = args.out_path + "custom/" + args.name + args.notes + "/"
    if rank == 0:
        if not os.path.exists(args_name_dir): os.makedirs(args_name_dir)
        args_name = args_name_dir + "/" + args.name +".yaml"
        if os.path.exists(args_name):
            s_add = 10
            logger.warning(f"Already exist args, add {s_add} to ran_seed to continue training")
            args.random_seed += s_add
        else:
            with open(args_name, "w+") as f:
                yaml.dump(args.__dict__, f, default_flow_style=True)
                #json.dump(args.__dict__, f)
    setup_logger(args_name_dir, rank, filename=f"{args.name}.txt")