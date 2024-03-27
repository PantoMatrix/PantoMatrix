import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pyarrow
import librosa
import smplx

from .build_vocab import Vocab
from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools_hf

class CustomDataset(Dataset):
    def __init__(self, args, loader_type, smplx_path=None, audio_path=None, text_path=None, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type
        self.smplx_path = "./EMAGE/test_sequences/smplxflame_30/2_scott_0_1_1.npz"
        self.audio_path = audio_path
        self.text_path = "./EMAGE/test_sequences/textgrid/2_scott_0_1_1.TextGrid"
        self.rank = 0 # dist.get_rank()
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0,0] # for trinity
        
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))  
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        # select trainable joints
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

        split_rule = pd.read_csv(args.data_path+"test.csv")
        self.selected_file = split_rule
        self.data_dir = args.data_path 
        
        if loader_type == "test": 
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length*self.args.audio_sr: 
            self.max_audio_pre_len = self.args.test_length*self.args.audio_sr
        
        if args.word_rep is not None:
            with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
                
        preloaded_dir = self.args.root_path + self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"        
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"] 
        

    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.args.disable_filtering, 
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)
        

    def __len__(self):
        return self.n_samples
    

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):      
        self.n_out_samples = 0
        # create db for samples
        if not os.path.exists(out_lmdb_dir): os.makedirs(out_lmdb_dir)
        if len(self.args.training_speakers) == 1:
            #dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 50))# 50G
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 0.5))# 500M
        else:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 200))# 200G
        n_filtered_out = defaultdict(int)
    
        #for index, file_name in self.selected_file.iterrows():
        #f_name = file_name["id"]
        ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
        pose_file = self.smplx_path#self.data_dir + self.args.pose_rep + "/" + f_name + ext
        pose_each_file = []
        trans_each_file = []
        shape_each_file = []
        audio_each_file = []
        facial_each_file = []
        word_each_file = []
        emo_each_file = []
        sem_each_file = []
        vid_each_file = []
        id_pose = "dummy 2nd"#f_name
        
        logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
        if "smplx" in self.args.pose_rep:
            pose_data = np.load(pose_file, allow_pickle=True)
            assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
            stride = int(30/self.args.pose_fps)
            pose_each_file = pose_data["poses"][::stride] 
            trans_each_file = pose_data["trans"][::stride]
            shape_each_file = np.repeat(pose_data["betas"].reshape(1, 300), pose_each_file.shape[0], axis=0)
            
            assert self.args.pose_fps == 30, "should 30"
            m_data = np.load(pose_file, allow_pickle=True)
            betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
            n, c = poses.shape[0], poses.shape[1]
            betas = betas.reshape(1, 300)
            betas = np.tile(betas, (n, 1))
            betas = torch.from_numpy(betas).cuda().float()
            poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
            exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
            trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
            max_length = 128
            s, r = n//max_length, n%max_length
            #print(n, s, r)
            all_tensor = []
            for i in range(s):
                with torch.no_grad():
                    joints = self.smplx(
                        betas=betas[i*max_length:(i+1)*max_length], 
                        transl=trans[i*max_length:(i+1)*max_length], 
                        expression=exps[i*max_length:(i+1)*max_length], 
                        jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                        global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                        body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                        left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                        right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                        reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                    )['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
                all_tensor.append(joints)
            if r != 0:
                with torch.no_grad():
                    joints = self.smplx(
                        betas=betas[s*max_length:s*max_length+r], 
                        transl=trans[s*max_length:s*max_length+r], 
                        expression=exps[s*max_length:s*max_length+r], 
                        jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                        global_orient=poses[s*max_length:s*max_length+r,:3], 
                        body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                        left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                        right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                        reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                    )['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
                all_tensor.append(joints)
            joints = torch.cat(all_tensor, axis=0) # all, 4, 3
            # print(joints.shape)
            feetv = torch.zeros(joints.shape[1], joints.shape[0])
            joints = joints.permute(1, 0, 2)
            #print(joints.shape, feetv.shape)
            feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
            #print(feetv.shape)
            contacts = (feetv < 0.01).numpy().astype(float)
            # print(contacts.shape, contacts)
            contacts = contacts.transpose(1, 0)
            pose_each_file = pose_each_file * self.joint_mask
            pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
            pose_each_file = np.concatenate([pose_each_file, contacts], axis=1)
            # print(pose_each_file.shape)
            
            
            if self.args.facial_rep is not None:
                logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                facial_each_file = pose_data["expressions"][::stride]
                if self.args.facial_norm: 
                    facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                
        else:
            assert 120%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 120'
            stride = int(120/self.args.pose_fps)
            with open(pose_file, "r") as pose_data:
                for j, line in enumerate(pose_data.readlines()):
                    if j < 431: continue     
                    if j%stride != 0:continue
                    data = np.fromstring(line, dtype=float, sep=" ")
                    rot_data = rc.euler_angles_to_matrix(torch.from_numpy(np.deg2rad(data)).reshape(-1, self.joints,3), "XYZ")
                    rot_data = rc.matrix_to_axis_angle(rot_data).reshape(-1, self.joints*3) 
                    rot_data = rot_data.numpy() * self.joint_mask
                    
                    pose_each_file.append(rot_data)
                    trans_each_file.append(data[:3])
                    
            pose_each_file = np.array(pose_each_file)
            trans_each_file = np.array(trans_each_file)
            shape_each_file = np.repeat(np.array(-1).reshape(1, 1), pose_each_file.shape[0], axis=0)
            if self.args.facial_rep is not None:
                logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                facial_file = pose_file.replace(self.args.pose_rep, self.args.facial_rep).replace("bvh", "json")
                assert 60%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 120'
                stride = int(60/self.args.pose_fps)
                if not os.path.exists(facial_file):
                    logger.warning(f"# ---- file not found for Facial {id_pose}, skip all files with the same id ---- #")
                    #self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                    #continue
                with open(facial_file, 'r') as facial_data_file:
                    facial_data = json.load(facial_data_file)
                    for j, frame_data in enumerate(facial_data['frames']):
                        if j%stride != 0:continue
                        facial_each_file.append(frame_data['weights'])
                facial_each_file = np.array(facial_each_file)
                if self.args.facial_norm: 
                    facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                    
        if self.args.id_rep is not None:
            int_value = 1
            vid_each_file = np.repeat(np.array(int_value).reshape(1, 1), pose_each_file.shape[0], axis=0)
    
        if self.args.audio_rep is not None:
            logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
            audio_file = self.audio_path[1]#pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
            sr = self.audio_path[0]
            #if not os.path.exists(audio_file):
            #    logger.warning(f"# ---- file not found for Audio  {id_pose}, skip all files with the same id ---- #")
                #self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                #continue
            #audio_each_file, sr = librosa.load(audio_file)
            audio_each_file = audio_file.astype(np.float32)
            audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=self.args.audio_sr)        
            if self.args.audio_rep == "onset+amplitude":
                from numpy.lib import stride_tricks
                frame_length = 1024
                # hop_length = 512
                shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
                strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
                rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
                amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                # pad the last frame_length-1 samples
                amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
                audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=self.args.audio_sr, units='frames')
                onset_array = np.zeros(len(audio_each_file), dtype=float)
                onset_array[audio_onset_f] = 1.0
                # print(amplitude_envelope.shape, audio_each_file.shape, onset_array.shape)
                audio_each_file = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
            elif self.args.audio_rep == "mfcc":
                audio_each_file = librosa.feature.melspectrogram(y=audio_each_file, sr=self.args.audio_sr, n_mels=128, hop_length=int(self.args.audio_sr/self.args.audio_fps))
                audio_each_file = audio_each_file.transpose(1, 0)
                # print(audio_each_file.shape, pose_each_file.shape)
            if self.args.audio_norm and self.args.audio_rep == "wave16k": 
                audio_each_file = (audio_each_file - self.mean_audio) / self.std_audio
                
        time_offset = 0
        if self.args.word_rep is not None:
            logger.info(f"# ---- Building cache for Word   {id_pose} and Pose {id_pose} ---- #")
            word_file = self.text_path#f"{self.data_dir}{self.args.word_rep}/{id_pose}.TextGrid"
            if not os.path.exists(word_file):
                logger.warning(f"# ---- file not found for Word   {id_pose}, skip all files with the same id ---- #")
                #self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                #continue
            tgrid = tg.TextGrid.fromFile(word_file)
            if self.args.t_pre_encoder == "bert":
                from transformers import AutoTokenizer, BertModel
                tokenizer = AutoTokenizer.from_pretrained(self.args.data_path_1 + "hub/bert-base-uncased", local_files_only=True)
                model = BertModel.from_pretrained(self.args.data_path_1 + "hub/bert-base-uncased", local_files_only=True).eval()
                list_word = []
                all_hidden = []
                max_len = 400
                last = 0
                word_token_mapping = []
                first = True
                for i, word in enumerate(tgrid[0]):
                    last = i
                    if (i%max_len != 0) or (i==0):
                        if word.mark == "":
                            list_word.append(".")
                        else:
                            list_word.append(word.mark)
                    else:
                        max_counter = max_len
                        str_word = ' '.join(map(str, list_word))
                        if first:
                            global_len = 0
                        end = -1
                        offset_word = []
                        for k, wordvalue in enumerate(list_word):
                            start = end+1 
                            end = start+len(wordvalue)
                            offset_word.append((start, end))
                        #print(offset_word)
                        token_scan = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
                        #print(token_scan)
                        for start, end in offset_word:
                            sub_mapping = []
                            for i, (start_t, end_t) in enumerate(token_scan[1:-1]):
                                if int(start) <= int(start_t) and int(end_t) <= int(end):
                                    #print(i+global_len)
                                    sub_mapping.append(i+global_len)
                            word_token_mapping.append(sub_mapping)
                        #print(len(word_token_mapping))
                        global_len = word_token_mapping[-1][-1] + 1    
                        list_word = []
                        if word.mark == "":
                            list_word.append(".")
                        else:
                            list_word.append(word.mark)
                        
                        with torch.no_grad():
                            inputs = tokenizer(str_word, return_tensors="pt")
                            outputs = model(**inputs)
                            last_hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
                        all_hidden.append(last_hidden_states)
                    
                #list_word = list_word[:10]
                if list_word == []:
                    pass
                else:
                    if first: 
                        global_len = 0
                    str_word = ' '.join(map(str, list_word))
                    end = -1
                    offset_word = []
                    for k, wordvalue in enumerate(list_word):
                        start = end+1 
                        end = start+len(wordvalue)
                        offset_word.append((start, end))
                    #print(offset_word)
                    token_scan = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
                    #print(token_scan)
                    for start, end in offset_word:
                        sub_mapping = []
                        for i, (start_t, end_t) in enumerate(token_scan[1:-1]):
                            if int(start) <= int(start_t) and int(end_t) <= int(end):
                                sub_mapping.append(i+global_len)
                                #print(sub_mapping)
                        word_token_mapping.append(sub_mapping)
                    #print(len(word_token_mapping))
                    with torch.no_grad():
                        inputs = tokenizer(str_word, return_tensors="pt")
                        outputs = model(**inputs)
                        last_hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
                    all_hidden.append(last_hidden_states)
                last_hidden_states = np.concatenate(all_hidden, axis=0)
        
            for i in range(pose_each_file.shape[0]):
                found_flag = False
                current_time = i/self.args.pose_fps + time_offset
                j_last = 0
                for j, word in enumerate(tgrid[0]): 
                    word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                    if word_s<=current_time and current_time<=word_e:
                        if self.args.word_cache and self.args.t_pre_encoder == 'bert':
                            mapping_index = word_token_mapping[j]
                            #print(mapping_index, word_s, word_e)
                            s_t = np.linspace(word_s, word_e, len(mapping_index)+1)
                            #print(s_t)
                            for tt, t_sep in enumerate(s_t[1:]):
                                if current_time <= t_sep:
                                    #if len(mapping_index) > 1: print(mapping_index[tt])
                                    word_each_file.append(last_hidden_states[mapping_index[tt]])
                                    break
                        else:
                            if word_n == " ":
                                word_each_file.append(self.lang_model.PAD_token)
                            else:
                                word_each_file.append(self.lang_model.get_word_index(word_n))
                        found_flag = True
                        j_last = j
                        break
                    else: continue   
                if not found_flag: 
                    if self.args.word_cache and self.args.t_pre_encoder == 'bert':
                        word_each_file.append(last_hidden_states[j_last])
                    else:
                        word_each_file.append(self.lang_model.UNK_token)
            word_each_file = np.array(word_each_file)
            #print(word_each_file.shape)
            
        if self.args.emo_rep is not None:
            logger.info(f"# ---- Building cache for Emo    {id_pose} and Pose {id_pose} ---- #")
            rtype, start = int(id_pose.split('_')[3]), int(id_pose.split('_')[3])
            if rtype == 0 or rtype == 2 or rtype == 4 or rtype == 6:
                if start >= 1 and start <= 64:
                    score = 0
                elif start >= 65 and start <= 72:
                    score = 1
                elif start >= 73 and start <= 80:
                    score = 2
                elif start >= 81 and start <= 86:
                    score = 3
                elif start >= 87 and start <= 94:
                    score = 4
                elif start >= 95 and start <= 102:
                    score = 5
                elif start >= 103 and start <= 110:
                    score = 6
                elif start >= 111 and start <= 118:
                    score = 7
                else: pass
            else:
                # you may denote as unknown in the future
                score = 0
            emo_each_file = np.repeat(np.array(score).reshape(1, 1), pose_each_file.shape[0], axis=0)    
            #print(emo_each_file)
            
        if self.args.sem_rep is not None:
            logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
            sem_file = f"{self.data_dir}{self.args.sem_rep}/{id_pose}.txt" 
            sem_all = pd.read_csv(sem_file, 
                sep='\t', 
                names=["name", "start_time", "end_time", "duration", "score", "keywords"])
            # we adopt motion-level semantic score here. 
            for i in range(pose_each_file.shape[0]):
                found_flag = False
                for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                    current_time = i/self.args.pose_fps + time_offset
                    if start<=current_time and current_time<=end: 
                        sem_each_file.append(score)
                        found_flag=True
                        break
                    else: continue 
                if not found_flag: sem_each_file.append(0.)
            sem_each_file = np.array(sem_each_file)
            #print(sem_each_file)
        
        filtered_result = self._sample_from_clip(
            dst_lmdb_env,
            audio_each_file, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
            vid_each_file, emo_each_file, sem_each_file,
            disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
            ) 
        for type in filtered_result.keys():
            n_filtered_out[type] += filtered_result[type]
                                
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
    
    def _sample_from_clip(
        self, dst_lmdb_env, audio_each_file, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        # audio_start = int(self.alignment[0] * self.args.audio_fps)
        # pose_start = int(self.alignment[1] * self.args.pose_fps)
        #logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        # audio_each_file = audio_each_file[audio_start:]
        # pose_each_file = pose_each_file[pose_start:]
        # trans_each_file = 
        #logger.info(f"after alignment: {audio_each_file.shape} {pose_each_file.shape}")
        #print(pose_each_file.shape)
        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # assume 1500 frames / 15 fps = 100 s
        #print(round_seconds_skeleton)
        if audio_each_file != []:
            if self.args.audio_rep != "wave16k":
                round_seconds_audio = len(audio_each_file) // self.args.audio_fps # assume 16,000,00 / 16,000 = 100 s
            elif self.args.audio_rep == "mfcc":
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_fps
            else:
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_sr
            if facial_each_file != []:
                round_seconds_facial = facial_each_file.shape[0] // self.args.pose_fps
                logger.info(f"audio: {round_seconds_audio}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
            else:
                logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.args.audio_fps * clip_s_t, clip_e_t * self.args.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps # [150,90*15]
        
        
        for ratio in self.args.multi_length_training:
            if is_test:# stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio*self.ori_stride)
                cut_length = int(self.ori_length*ratio)
                
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")
            
            if audio_each_file != []:
                audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
                """
                for audio sr = 16000, fps = 15, pose_length = 34, 
                audio short length = 36266.7 -> 36266
                this error is fine.
                """
                logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")
             
            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_audio_list = []
            sample_facial_list = []
            sample_shape_list = []
            sample_word_list = []
            sample_emo_list = []
            sample_sem_list = []
            sample_vid_list = []
            sample_trans_list = []
           
            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length 
                sample_pose = pose_each_file[start_idx:fin_idx]

                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)
                if self.args.audio_rep is not None:
                    audio_start = clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps)
                    audio_end = audio_start + audio_short_length
                    sample_audio = audio_each_file[audio_start:audio_end]
                else:
                    sample_audio = np.array([-1])
                sample_facial = facial_each_file[start_idx:fin_idx] if self.args.facial_rep is not None else np.array([-1])
                sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
                sample_emo = emo_each_file[start_idx:fin_idx] if self.args.emo_rep is not None else np.array([-1])
                sample_sem = sem_each_file[start_idx:fin_idx] if self.args.sem_rep is not None else np.array([-1])
                sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None else np.array([-1])
                
                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    is_correct_motion = (sample_pose != [])
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_audio_list.append(sample_audio)
                        sample_facial_list.append(sample_facial)
                        sample_shape_list.append(sample_shape)
                        sample_word_list.append(sample_word)
                        sample_vid_list.append(sample_vid)
                        sample_emo_list.append(sample_emo)
                        sample_sem_list.append(sample_sem)
                        sample_trans_list.append(sample_trans)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, audio, facial, shape, word, vid, emo, sem, trans in zip(
                        sample_pose_list,
                        sample_audio_list,
                        sample_facial_list,
                        sample_shape_list,
                        sample_word_list,
                        sample_vid_list,
                        sample_emo_list,
                        sample_sem_list,
                        sample_trans_list,):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [pose, audio, facial, shape, word, emo, sem, vid, trans]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans = sample
            #print(in_shape)
            #vid = torch.from_numpy(vid).int()
            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float() 
            in_audio = torch.from_numpy(in_audio).float() 
            in_word = torch.from_numpy(in_word).float() if self.args.word_cache else torch.from_numpy(in_word).int() 
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                trans = torch.from_numpy(trans).float()
                in_facial = torch.from_numpy(in_facial).float()
                vid = torch.from_numpy(vid).float()
                in_shape = torch.from_numpy(in_shape).float()
            else:
                in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            return {"pose":tar_pose, "audio":in_audio, "facial":in_facial, "beta": in_shape, "word":in_word, "id":vid, "emo":emo, "sem":sem, "trans":trans}

         
class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        #self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
#         diff = np.abs(self.skeletons - self.mean_pose) # 186*1
#         diff = np.mean(diff)

#         # th = 0.017
#         th = 0.02 #0.02  # exclude 3594
#         if diff < th:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
# #         th = 3.5 #0.02  # exclude 3594
# #         if 3.5 < diff < 5:
# #             if verbose:
# #                 print("skip - check_pose_diff {:.5f}".format(diff))
# #             return True
#         else:
#             if verbose:
#                 print("pass - check_pose_diff {:.5f}".format(diff))
        return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False
