'''
content
0 1269 0.13
1 1282 0.14
2 1142 0.12
3 974 0.1
4 859 0.09
5 519 0.05
6 909 0.1
7 819 0.09
8 803 0.08
9 909 0.1

rhythm
0 798 0.08
1 1195 0.12
2 1007 0.1
3 831 0.08
4 1153 0.12
5 1228 0.12
6 985 0.1
7 852 0.09
8 1139 0.12
9 654 0.07
'''

import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from emage_utils.motion_io import beat_format_load, MASK_DICT

class BEAT2DatasetDisco(Dataset):
    def __init__(self, cfg, split):
        vid_meta = []
        for data_meta_path in cfg.data.meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = [i for i in vid_meta if i.get("mode") == split]
        self.mean = 0
        self.std = 1
        self.joint_mask = MASK_DICT[cfg.model.joint_mask] if cfg.model.joint_mask else None
        self.data_list = self.vid_meta
        self.fps = cfg.model.pose_fps
        self.audio_sr = cfg.model.audio_sr

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def normalize(motion, mean, std):
        return (motion - mean) / (std + 1e-7)
    
    @staticmethod
    def inverse_normalize(motion, mean, std):
        return motion * std + mean

    def __getitem__(self, item):
        data_item = self.data_list[item]
        smplx_data = beat_format_load(data_item["motion_path"], mask=self.joint_mask)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        motion = smplx_data["poses"][sdx:edx]
        SMPLX_FPS = 30
        downsample_factor = SMPLX_FPS // self.fps
        motion = motion[::downsample_factor]
        motion = self.normalize(motion, self.mean, self.std)
        
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = sdx * int((1 / SMPLX_FPS) * self.audio_sr)
        edx_audio = edx * int((1 / SMPLX_FPS) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
             
        content_label = torch.tensor(data_item["content_label"]).long()
        rhythm_label = torch.tensor(data_item["rhythm_label"]).long()

        return dict(
            motion=torch.from_numpy(motion).float(),
            audio=torch.from_numpy(audio).float(),
            content_label=content_label,
            rhythm_label=rhythm_label
        )