import json
import torch
from torch.utils import data
import numpy as np
import librosa
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from emage_utils.motion_io import beat_format_load, MASK_DICT

class BEAT2Dataset(data.Dataset):
    def __init__(self, cfg, split):
        vid_meta = []
        for data_meta_path in cfg.data.meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = [item for item in vid_meta if item.get("mode") == split]
        self.mean = 0
        self.std = 1
        self.joint_mask = MASK_DICT[cfg.model.joint_mask] if cfg.model.joint_mask is not None else None
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
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
       
        return dict(
            motion=motion_tensor,
            audio=audio_tensor, 
        )

class BEAT2DatasetEamge(BEAT2Dataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

    def __getitem__(self, item):
        data_item = self.data_list[item]
        smplx_data = beat_format_load(data_item["motion_path"], mask=None)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        motion = smplx_data["poses"][sdx:edx]
        expressions = smplx_data["expressions"][sdx:edx]
        trans = smplx_data["trans"][sdx:edx]
        SMPLX_FPS = 30
        downsample_factor = SMPLX_FPS // self.fps
        motion = motion[::downsample_factor]
        motion = self.normalize(motion, self.mean, self.std)
        
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = sdx * int((1 / SMPLX_FPS) * self.audio_sr)
        edx_audio = edx * int((1 / SMPLX_FPS) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
        expressions_tesnor = torch.from_numpy(expressions).float()
        trans_tensor = torch.from_numpy(trans).float()

        return dict(
            motion=motion_tensor,
            audio=audio_tensor, 
            expressions=expressions_tesnor,
            trans=trans_tensor,
        )


class BEAT2DatasetEamgeFootContact(BEAT2Dataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

    def __getitem__(self, item):
        data_item = self.data_list[item]
        smplx_data = beat_format_load(data_item["motion_path"], mask=None)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        motion = smplx_data["poses"][sdx:edx]
        expressions = smplx_data["expressions"][sdx:edx]
        trans = smplx_data["trans"][sdx:edx]
        foot_contact = np.load(data_item["motion_path"].replace("smplxflame_30", "footcontact").replace(".npz", ".npy"))[sdx:edx]

        SMPLX_FPS = 30
        downsample_factor = SMPLX_FPS // self.fps
        motion = motion[::downsample_factor]
        motion = self.normalize(motion, self.mean, self.std)
        
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = sdx * int((1 / SMPLX_FPS) * self.audio_sr)
        edx_audio = edx * int((1 / SMPLX_FPS) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
        expressions_tesnor = torch.from_numpy(expressions).float()
        trans_tensor = torch.from_numpy(trans).float()
        foot_contact_tensor = torch.from_numpy(foot_contact).float()
        # print(trans_tensor.shape, foot_contact_tensor.shape)

        return dict(
            motion=motion_tensor,
            audio=audio_tensor, 
            expressions=expressions_tesnor,
            trans=trans_tensor,
            foot_contact=foot_contact_tensor,
        )
