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
from sklearn.preprocessing import normalize
# import librosa 
import scipy.io.wavfile
from scipy import signal
from .build_vocab import Vocab


class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.loader_type = loader_type
        self.rank = dist.get_rank()
        self.new_cache = args.new_cache
        self.pose_length = args.pose_length #34
        self.stride = args.stride #10
        self.pose_fps = args.pose_fps #15
        self.pose_dims = args.pose_dims # 141

        self.speaker_dims = args.speaker_dims
        self.loader_type = loader_type
        self.audio_rep = args.audio_rep
        self.pose_rep = args.pose_rep
        self.facial_rep = args.facial_rep
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.audio_fps = args.audio_fps
        self.id_rep = args.speaker_id
        
        self.disable_filtering = args.disable_filtering
        self.clean_first_seconds = args.clean_first_seconds
        self.clean_final_seconds = args.clean_final_seconds
        
        self.ori_stride = self.stride
        self.ori_length = self.pose_length
        self.alignment = [0,0] # for beat
        
        if loader_type == "train":
            self.data_dir = args.root_path + args.train_data_path
            self.multi_length_training = args.multi_length_training
        elif loader_type == "val":
            self.data_dir = args.root_path + args.val_data_path
            self.multi_length_training = args.multi_length_training 
        else:
            self.data_dir = args.root_path + args.test_data_path
            self.multi_length_training = [1.0]
      
        self.max_length = int(self.pose_length * self.multi_length_training[-1])
        
        if self.word_rep is not None:
            with open(f"{args.root_path}{args.train_data_path[:-6]}vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
        preloaded_dir = self.data_dir + f"{self.pose_rep}_cache"
        
        self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
        self.audio_norm = args.audio_norm
        self.facial_norm = args.facial_norm
        if self.audio_norm:
            self.mean_audio = np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_mean.npy")
            self.std_audio = np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_std.npy")
        if self.facial_norm:
            self.mean_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")
            self.std_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy")
            
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]    

            
    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        
        # pose_length_extended = int(round(self.pose_length))
        logger.info("Creating the dataset cache...")
        if self.new_cache:
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
                preloaded_dir, self.disable_filtering, 
                self.clean_first_seconds, self.clean_final_seconds,
                is_test=False)
        
    
    def __len__(self):
        return self.n_samples

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        pose_files = sorted(glob.glob(os.path.join(self.data_dir, f"{self.pose_rep}") + "/*.bvh"), key=str,)  
        # create db for samples
        map_size = int(1024 * 1024 * 2048 * (self.audio_fps/16000)**3 * 4) * (len(pose_files)/30*(self.pose_fps/15)) * len(self.multi_length_training) * self.multi_length_training[-1] * 2 # in 1024 MB
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        n_filtered_out = defaultdict(int)
    
        for pose_file in pose_files:
            pose_each_file = []
            audio_each_file = []
            facial_each_file = []
            word_each_file = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = []
            
            id_pose = pose_file.split("/")[-1][:-4] #1_wayne_0_1_1
            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            with open(pose_file, "r") as pose_data:
                for j, line in enumerate(pose_data.readlines()):
                    data = np.fromstring(line, dtype=float, sep=" ") # 1*27 e.g., 27 rotation 
                    pose_each_file.append(data)
            pose_each_file = np.array(pose_each_file) # n frames * 27

            if self.audio_rep is not None:
                logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
                audio_file = pose_file.replace(self.pose_rep, self.audio_rep).replace("bvh", "npy")
                try:
                    # the librosa cannot use on the cloud sever
#                     audio_data, _ = librosa.load(audio_file, sr=None)
#                     if self.audio_rep == "melspec":
#                         audio_each_file = np.load(f"{audio_file[:-4]}_melspec_128_64.npy").transpose(1,0)
#                         self.audio_fps = 32
#                     elif self.audio_rep == "disentangled":
#                         audio_each_file = np.load(f"{audio_file[:-4]}_disentangled_v1.npy").transpose(1,0)
#                     else:
#                         sr, audio_each_file = scipy.io.wavfile.read(audio_file) # np array
#                     audio_each_file = audio_each_file[::sr//16000]
                    audio_each_file = np.load(audio_file)
                except:
                    logger.warning(f"# ---- file not found for Audio {id_pose}, skip all files with the same id ---- #")
                    continue
                if self.audio_norm: 
                    audio_each_file = (audio_each_file - self.mean_audio) / self.std_audio
                    
            if self.facial_rep is not None:
                logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                facial_file = pose_file.replace(self.pose_rep, self.facial_rep).replace("bvh", "json")
                try:
                    with open(facial_file, 'r') as facial_data_file:
                        facial_data = json.load(facial_data_file)
                        for j, frame_data in enumerate(facial_data['frames']):
                            if self.facial_norm:
                                facial_each_file.append((frame_data['weights']-self.mean_facial) / self.std_facial)
                            else:
                                facial_each_file.append(frame_data['weights'])
                    facial_each_file = np.array(facial_each_file)
                except:
                    logger.warning(f"# ---- file not found for Facial {id_pose}, skip all files with the same id ---- #")
                    continue
                    
            if id_pose.split("_")[-1] == "b":
                time_offset = 30 if int(id_pose.split("_")[-3]) % 2 == 0 else 300
                logger.warning(time_offset)
            else:
                time_offset = 0
                
            if self.word_rep is not None:
                logger.info(f"# ---- Building cache for Word   {id_pose} and Pose {id_pose} ---- #")
                word_file = pose_file.replace(self.pose_rep, self.word_rep).replace("bvh", "TextGrid")
                try:
                    tgrid = tg.TextGrid.fromFile(word_file)
                except:
                    logger.warning(f"# ---- file not found for Word {id_pose}, skip all files with the same id ---- #")
                    continue
                # the length of text file are reduce to the length of motion file, for x_x_a or x_x_b
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    current_time = i/self.pose_fps + time_offset
                    for word in tgrid[0]:
                        word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                        if word_s<=current_time and current_time<=word_e:
                            if word_n == " ":
                                #TODO now don't have eos and sos token
                                word_each_file.append(self.lang_model.PAD_token)
                            else:    
                                word_each_file.append(self.lang_model.get_word_index(word_n))
                            found_flag = True
                            break
                        else: continue   
                    if not found_flag: word_each_file.append(self.lang_model.UNK_token)
                # list of index
                word_each_file = np.array(word_each_file)
                    
            if self.emo_rep is not None:
                logger.info(f"# ---- Building cache for Emo    {id_pose} and Pose {id_pose} ---- #")
                emo_file = pose_file.replace(self.pose_rep, self.emo_rep).replace("bvh", "csv")
                try:    
                    emo_all = pd.read_csv(emo_file, 
                        sep=',', 
                        names=["name", "start_time", "end_time", "duration", "score"])
                except:
                    logger.warning(f"# ---- file not found for Emo {id_pose}, skip all files with the same id ---- #")
                    continue
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(emo_all['start_time'],emo_all['end_time'], emo_all['score'])):
                        current_time = i/self.pose_fps + time_offset
                        if start<=current_time and current_time<=end: 
                            emo_each_file.append(score)
                            found_flag=True
                            break
                        else: continue 
                    if not found_flag: emo_each_file.append(0)
                emo_each_file = np.array(emo_each_file)
                #print(emo_each_file)
                
            if self.sem_rep is not None:
                logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
                sem_file = pose_file.replace(self.pose_rep, self.sem_rep).replace("bvh", "txt")
                try:
                    sem_all = pd.read_csv(sem_file, 
                        sep='\t', 
                        names=["name", "start_time", "end_time", "duration", "score", "keywords"])
                except:
                    logger.warning(f"# ---- file not found for Sem {id_pose}, skip all files with the same id ---- #")
                    continue
                # we adopt motion-level semantic score here. 
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                        current_time = i/self.pose_fps + time_offset
                        if start<=current_time and current_time<=end: 
                            sem_each_file.append(score)
                            found_flag=True
                            break
                        else: continue 
                    if not found_flag: sem_each_file.append(0.)
                sem_each_file = np.array(sem_each_file)
                #print(sem_each_file)
                
            if self.id_rep is not None:
                vid_each_file.append(int(id_pose.split("_")[0])-1)
            
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_each_file, facial_each_file, word_each_file,
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
        self, dst_lmdb_env, audio_each_file, pose_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        audio_start = int(self.alignment[0] * self.audio_fps)
        pose_start = int(self.alignment[1] * self.pose_fps)
        #logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        audio_each_file = audio_each_file[audio_start:]
        pose_each_file = pose_each_file[pose_start:]
        #logger.info(f"after alignment: {audio_each_file.shape} {pose_each_file.shape}")
        
        round_seconds_skeleton = pose_each_file.shape[0] // self.pose_fps  # assume 1500 frames / 15 fps = 100 s
        if audio_each_file != []:
            round_seconds_audio = len(audio_each_file) // self.audio_fps # assume 16,000,00 / 16,000 = 100 s
            if facial_each_file != []:
                round_seconds_facial = facial_each_file.shape[0] // self.pose_fps
                logger.info(f"audio: {round_seconds_skeleton}s, pose: {round_seconds_audio}s, facial: {round_seconds_facial}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
            else:
                logger.info(f"audio: {round_seconds_skeleton}s, pose: {round_seconds_audio}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.audio_fps * clip_s_t, clip_e_t * self.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.pose_fps, clip_e_t * self.pose_fps # [150,90*15]

        for ratio in self.multi_length_training:
            if is_test:# stride = length for test
                self.pose_length = clip_e_f_pose - clip_s_f_pose
                self.stride = self.pose_length
                self.max_length = self.pose_length
            else:
                self.stride = int(ratio*self.ori_stride)
                self.pose_length = int(self.ori_length*ratio)
                
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - self.pose_length) / self.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {self.pose_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.stride}")
            
            if audio_each_file != []:
                audio_short_length = math.floor(self.pose_length / self.pose_fps * self.audio_fps)
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
            sample_word_list = []
            sample_emo_list = []
            sample_sem_list = []
            sample_vid_list = []
           
            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.stride
                fin_idx = start_idx + self.pose_length # 34
                sample_pose = pose_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)
                if audio_each_file != []:
                    audio_start = clip_s_f_audio + math.floor(i * self.stride * self.audio_fps / self.pose_fps)
                    audio_end = audio_start + audio_short_length
                    sample_audio = audio_each_file[audio_start:audio_end]
                elif self.audio_rep is not None:
                    logger.warning("audio file is incorrect")
                    continue
                else:
                    sample_audio = np.array([-1])
                
                sample_facial = facial_each_file[start_idx:fin_idx] if facial_each_file != [] else np.array([-1])
                sample_word = word_each_file[start_idx:fin_idx] if word_each_file != [] else np.array([-1])
                sample_emo = emo_each_file[start_idx:fin_idx] if emo_each_file != [] else np.array([-1])
                sample_sem = sem_each_file[start_idx:fin_idx] if sem_each_file != [] else np.array([-1])
                sample_vid = np.array(vid_each_file) if vid_each_file != [] else np.array([-1])
                
                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose, self.mean_pose).get()
                    is_correct_motion = (sample_pose != [])
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_audio_list.append(sample_audio)
                        sample_facial_list.append(sample_facial)
                        sample_word_list.append(sample_word)
                        sample_vid_list.append(sample_vid)
                        sample_emo_list.append(sample_emo)
                        sample_sem_list.append(sample_sem)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, audio, facial, word, vid, emo, sem in zip(sample_pose_list,
                                                        sample_audio_list,
                                                        sample_facial_list,
                                                        sample_word_list,
                                                        sample_vid_list,
                                                        sample_emo_list,
                                                        sample_sem_list,
                                                        ):
                        normalized_pose = self.normalize_pose(pose, self.mean_pose, self.std_pose)
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [normalized_pose, audio, facial, word, emo, sem, vid]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    @staticmethod
    def normalize_pose(dir_vec, mean_pose, std_pose=None):
        return (dir_vec - mean_pose) / std_pose 
    
    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            tar_pose, in_audio, in_facial, in_word, emo, sem, vid = sample
            vid = torch.from_numpy(vid).int()
            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float() 
            in_audio = torch.from_numpy(in_audio).float() 
            in_word = torch.from_numpy(in_word).int()  
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                in_facial = torch.from_numpy(in_facial).float()
                            
            else:
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            return {"pose":tar_pose, "audio":in_audio, "facial":in_facial, "word":in_word, "id":vid, "emo":emo, "sem":sem}

         
class MotionPreprocessor:
    def __init__(self, skeletons, mean_pose):
        self.skeletons = skeletons
        self.mean_pose = mean_pose
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
        diff = np.abs(self.skeletons - self.mean_pose) # 186*1
        diff = np.mean(diff)

        # th = 0.017
        th = 0.02 #0.02  # exclude 3594
        if diff < th:
            if verbose:
                print("skip - check_pose_diff {:.5f}".format(diff))
            return True
#         th = 3.5 #0.02  # exclude 3594
#         if 3.5 < diff < 5:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
        else:
            if verbose:
                print("pass - check_pose_diff {:.5f}".format(diff))
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