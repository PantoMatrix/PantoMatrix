import configargparse
import time
import json
import yaml
import os


def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')
        

def parse_args():
    """
    requirement for config
    1. command > yaml > default
    2. avoid re-definition 
    3. lowercase letters is better
    4. hierarchical is not necessary
    """
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", default='./scripts/EMAGE_2024/configs/emage_test_hf.yaml', is_config_file=True)
    parser.add("--project", default="audio2pose", type=str) # wandb project name
    parser.add("--stat", default="ts", type=str)
    parser.add("--csv_name", default="a2g_0", type=str) # local device id
    parser.add("--notes", default="", type=str) 
    parser.add("--trainer", default="camn", type=str) 

    parser.add("--l", default=4, type=int)
    # ------------- path and save name ---------------- #
    parser.add("--is_train", default=True, type=str2bool)
    parser.add("--debug", default=False, type=str2bool)
    # different between environments
    parser.add("--root_path", default="/home/ma-user/work/")
    parser.add("--cache_path", default="/outputs/audio2pose/", type=str)
    parser.add("--out_path", default="/outputs/audio2pose/", type=str)
    parser.add("--data_path", default="/outputs/audio2pose/", type=str)
    parser.add("--train_data_path", default="/datasets/trinity/train/", type=str)
    parser.add("--val_data_path", default="/datasets/trinity/val/", type=str)
    parser.add("--test_data_path", default="/datasets/trinity/test/", type=str)
    parser.add("--mean_pose_path", default="/datasets/trinity/train/", type=str)
    parser.add("--std_pose_path", default="/datasets/trinity/train/", type=str)
    # for pretrian weights
    parser.add("--data_path_1", default="../../datasets/checkpoints/", type=str)
    # ------------------- evaluation ----------------------- #
    parser.add("--test_ckpt", default="/datasets/beat_cache/beat_4english_15_141/last.bin")
    parser.add("--eval_model", default="vae", type=str)
    parser.add("--e_name", default=None, type=str) #HalfEmbeddingNet
    parser.add("--e_path", default="/datasets/beat/generated_data/self_vae_128.bin")
    parser.add("--variational", default=False, type=str2bool) 
    parser.add("--vae_length", default=256, type=int)
    parser.add("--vae_test_dim", default=141, type=int)
    parser.add("--vae_test_len", default=34, type=int)
    parser.add("--vae_test_stride", default=10, type=int)
    #parser.add("--vae_pose_length", default=34, type=int)
    parser.add("--test_period", default=20, type=int)
    parser.add("--vae_codebook_size", default=1024, type=int)
    parser.add("--vae_quantizer_lambda", default=1., type=float)
    
    parser.add("--vae_layer", default=2, type=int)
    parser.add("--vae_grow", default=[1,1,2,1], type=int, nargs="*")
    parser.add("--lf", default=0., type=float)
    parser.add("--ll", default=0., type=float)
    parser.add("--lu", default=0., type=float)
    parser.add("--lh", default=0., type=float)
    parser.add("--cf", default=0., type=float)
    parser.add("--cl", default=0., type=float)
    parser.add("--cu", default=0., type=float)
    parser.add("--ch", default=0., type=float)
    
    
    # --------------- data ---------------------------- #
    parser.add("--additional_data", default=False, type=str2bool)
    parser.add("--train_trans", default=True, type=str2bool)
    parser.add("--dataset", default="beat", type=str)
    parser.add("--rot6d", default=True, type=str2bool)
    parser.add("--ori_joints", default="spine_neck_141", type=str)
    parser.add("--tar_joints", default="spine_neck_141", type=str)
    parser.add("--training_speakers", default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], type=int, nargs="*")
    #parser.add("--pose_version", default="spine_neck_141", type=str)
    parser.add("--new_cache", default=True, type=str2bool)
    parser.add("--beat_align", default=True, type=str2bool)
    parser.add("--cache_only", default=False, type=str2bool)
    parser.add("--word_cache", default=False, type=str2bool)
    parser.add("--use_aug", default=False, type=str2bool)
    parser.add("--disable_filtering", default=False, type=str2bool)
    parser.add("--clean_first_seconds", default=0, type=int)
    parser.add("--clean_final_seconds", default=0, type=int)

    parser.add("--audio_rep", default=None, type=str)
    parser.add("--audio_sr", default=16000, type=int)
    parser.add("--word_rep", default=None, type=str)
    parser.add("--emo_rep", default=None, type=str)
    parser.add("--sem_rep", default=None, type=str)
    parser.add("--facial_rep", default=None, type=str)
    parser.add("--pose_rep", default="bvhrot", type=str)
    parser.add("--id_rep", default="onehot", type=str)
    parser.add("--speaker_id", default="onehot", type=str)
    
    parser.add("--a_pre_encoder", default=None, type=str)
    parser.add("--a_encoder", default=None, type=str)
    parser.add("--a_fix_pre", default=False, type=str2bool)
    parser.add("--t_pre_encoder", default=None, type=str)
    parser.add("--t_encoder", default=None, type=str)
    parser.add("--t_fix_pre", default=False, type=str2bool)
    parser.add("--m_pre_encoder", default=None, type=str)
    parser.add("--m_encoder", default=None, type=str)
    parser.add("--m_fix_pre", default=False, type=str2bool)
    parser.add("--f_pre_encoder", default=None, type=str)
    parser.add("--f_encoder", default=None, type=str)
    parser.add("--f_fix_pre", default=False, type=str2bool)
    parser.add("--m_decoder", default=None, type=str)
    parser.add("--decode_fusion", default=None, type=str)
    parser.add("--atmr", default=0.0, type=float)
    parser.add("--ttmr", default=0., type=float)
    parser.add("--mtmr", default=0., type=float)
    parser.add("--ftmr", default=0., type=float)
    parser.add("--asmr", default=0., type=float)
    parser.add("--tsmr", default=0., type=float)
    parser.add("--msmr", default=0., type=float)
    parser.add("--fsmr", default=0., type=float)
#    parser.add("--m_encoder", default=None, type=str)
#    parser.add("--m_pre_fix", default=None, type=str)
#    parser.add("--id_rep", default=None, type=str)
    
    parser.add("--freeze_wordembed", default=True, type=str2bool)
    parser.add("--audio_fps", default=16000, type=int)
    parser.add("--facial_fps", default=15, type=int)
    parser.add("--pose_fps", default=15, type=int)
    
    parser.add("--audio_dims", default=1, type=int)
    parser.add("--facial_dims", default=39, type=int)
    parser.add("--pose_dims", default=123, type=int)
    parser.add("--word_index_num", default=5793, type=int)
    parser.add("--word_dims", default=300, type=int)
    parser.add("--speaker_dims", default=4, type=int)
    parser.add("--emotion_dims", default=8, type=int)
    
    parser.add("--audio_norm", default=False, type=str2bool)
    parser.add("--facial_norm", default=False, type=str2bool)
    parser.add("--pose_norm", default=False, type=str2bool)
        
    parser.add("--pose_length", default=34, type=int)
    parser.add("--pre_frames", default=4, type=int)
    parser.add("--stride", default=10, type=int)
    parser.add("--pre_type", default="zero", type=str)
    
    parser.add("--audio_f", default=0, type=int)
    parser.add("--motion_f", default=0, type=int)
    parser.add("--facial_f", default=0, type=int)
    parser.add("--speaker_f", default=0, type=int)
    parser.add("--word_f", default=0, type=int)
    parser.add("--emotion_f", default=0, type=int)
    parser.add("--aud_prob", default=1.0, type=float)
    parser.add("--pos_prob", default=1.0, type=float)
    parser.add("--txt_prob", default=1.0, type=float)
    parser.add("--fac_prob", default=1.0, type=float)
    parser.add("--multi_length_training", default=[1.0], type=float, nargs="*")
    # --------------- model ---------------------------- #
    parser.add("--pretrain", default=False, type=str2bool)
    parser.add("--model", default="camn", type=str)
    parser.add("--g_name", default="CaMN", type=str)
    parser.add("--d_name", default=None, type=str) #ConvDiscriminator
    parser.add("--dropout_prob", default=0.3, type=float)
    parser.add("--n_layer", default=4, type=int)
    parser.add("--hidden_size", default=300, type=int)
    #parser.add("--period", default=34, type=int)
    parser.add("--test_length", default=34, type=int)
    # Self-designed "Multi-Stage", "Seprate", or "Original"
    parser.add("--finger_net", default="original", type=str)
    parser.add("--pos_encoding_type", default="sin", type=str)
    parser.add("--queue_size", default=1024, type=int)
    
    # --------------- training ------------------------- #
    parser.add("--epochs", default=120, type=int)
    parser.add("--epoch_stage", default=0, type=int)
    parser.add("--grad_norm", default=0, type=float)
    parser.add("--no_adv_epoch", default=999, type=int)
    parser.add("--batch_size", default=128, type=int)
    parser.add("--opt", default="adam", type=str)
    parser.add("--lr_base", default=0.00025, type=float)
    parser.add("--opt_betas", default=[0.5, 0.999], type=float, nargs="*")
    parser.add("--weight_decay", default=0., type=float)
    # for warmup and cosine
    parser.add("--lr_min", default=1e-7, type=float)
    parser.add("--warmup_lr", default=5e-4, type=float)
    parser.add("--warmup_epochs", default=0, type=int)
    parser.add("--decay_epochs", default=9999, type=int)
    parser.add("--decay_rate", default=0.1, type=float)
    parser.add("--lr_policy", default="step", type=str)
    # for sgd
    parser.add("--momentum", default=0.8, type=float)
    parser.add("--nesterov", default=True, type=str2bool)
    parser.add("--amsgrad", default=False, type=str2bool)
    parser.add("--d_lr_weight", default=0.2, type=float)
    parser.add("--rec_weight", default=500, type=float)
    parser.add("--adv_weight", default=20.0, type=float)
    parser.add("--fid_weight", default=0.0, type=float)
    parser.add("--vel_weight", default=0.0, type=float)
    parser.add("--acc_weight", default=0.0, type=float)
    parser.add("--kld_weight", default=0.0, type=float)
    parser.add("--kld_aud_weight", default=0.0, type=float)
    parser.add("--kld_fac_weight", default=0.0, type=float)
    parser.add("--ali_weight", default=0.0, type=float)
    parser.add("--ita_weight", default=0.0, type=float)
    parser.add("--iwa_weight", default=0.0, type=float)
    parser.add("--wei_weight", default=0.0, type=float)
    parser.add("--gap_weight", default=0.0, type=float)
    parser.add("--atcont", default=0.0, type=float)
    parser.add("--fusion_mode", default="sum", type=str)
    
    parser.add("--div_reg_weight", default=0.0, type=float)
    parser.add("--rec_aud_weight", default=0.0, type=float)
    parser.add("--rec_ver_weight", default=0.0, type=float)
    parser.add("--rec_pos_weight", default=0.0, type=float)
    parser.add("--rec_fac_weight", default=0.0, type=float)
    parser.add("--rec_txt_weight", default=0.0, type=float)
#    parser.add("--gan_noise_size", default=0, type=int)
    # --------------- ha2g -------------------------- #
    parser.add("--n_pre_poses", default=4, type=int)
    parser.add("--n_poses", default=34, type=int)
    parser.add("--input_context", default="both", type=str)
    parser.add("--loss_contrastive_pos_weight", default=0.2, type=float)
    parser.add("--loss_contrastive_neg_weight", default=0.005, type=float)
    parser.add("--loss_physical_weight", default=0.0, type=float)
    parser.add("--loss_reg_weight", default=0.05, type=float)
    parser.add("--loss_regression_weight", default=70.0, type=float)
    parser.add("--loss_gan_weight", default=5.0, type=float)
    parser.add("--loss_kld_weight", default=0.1, type=float)
    parser.add("--z_type", default="speaker", type=str)
    # --------------- device -------------------------- #
    parser.add("--random_seed", default=2021, type=int)
    parser.add("--deterministic", default=True, type=str2bool)
    parser.add("--benchmark", default=True, type=str2bool)
    parser.add("--cudnn_enabled", default=True, type=str2bool)
    # mix precision
    parser.add("--apex", default=False, type=str2bool)
    parser.add("--gpus", default=[0], type=int, nargs="*")
    parser.add("--loader_workers", default=0, type=int)
    parser.add("--ddp", default=False, type=str2bool)
    parser.add("--sparse", default=1, type=int)
    #parser.add("--world_size")
    parser.add("--render_video_fps", default=30, type=int)
    parser.add("--render_video_width", default=1920, type=int)
    parser.add("--render_video_height", default=720, type=int)
    cpu_cores = os.cpu_count() if os.cpu_count() is not None else 1
    default_concurrent = max(1, cpu_cores // 2)
    parser.add("--render_concurrent_num", default=default_concurrent, type=int)
    parser.add("--render_tmp_img_filetype", default="bmp", type=str)
    
    # logging
    parser.add("--log_period", default=10, type=int)
  
    
    args = parser.parse_args()
    idc = 0
    for i, char in enumerate(args.config):
        if char == "/": idc = i
    args.name = args.config[idc+1:-5]
    
    is_train = args.is_train

    if is_train:
        time_local = time.localtime()
        name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        args.name = name_expend + args.name
        
    return args