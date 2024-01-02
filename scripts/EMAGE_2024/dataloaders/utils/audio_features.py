"""modified from https://github.com/yesheng-THU/GFGE/blob/main/data_processing/audio_features.py"""
import numpy as np
import librosa
import math
import os
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple
_CONFIG_FOR_DOC = "Wav2Vec2Config"

# the implementation of Wav2Vec2Model is borrowed from https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
# initialize our encoder with the pre-trained wav2vec 2.0 weights.
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask

# linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)

class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.audio_fps = 15 #args.audio_fps
        #input_values 16K hz, 49fps, 20ms overlap, 25ms recepion field 
    def forward(
        self,
        input_values,
        dataset="beat",
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        frame_num=None
    ):  
        #print(input_values.shape)
        self.config.output_attentions = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)
        #print(hidden_states.shape)
        if dataset == "beat":
            hidden_states = linear_interpolation(hidden_states, 49, self.audio_fps, output_len=frame_num)
        #print(hidden_states.shape)
        if attention_mask is not None:
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            attention_mask = torch.zeros(
                hidden_states.shape[:2], dtype=hidden_states.dtype, device=hidden_states.device
            )
            attention_mask[
                (torch.arange(attention_mask.shape[0], device=hidden_states.device), output_lengths - 1)
            ] = 1
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        hidden_states = self.feature_projection(hidden_states)[0]
        #print(hidden_states.shape)
        if self.config.apply_spec_augment and self.training:
            batch_size, sequence_length, hidden_size = hidden_states.size()
            if self.config.mask_time_prob > 0:
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    self.config.mask_time_prob,
                    self.config.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=2,
                )
                hidden_states[torch.from_numpy(mask_time_indices)] = self.masked_spec_embed.to(hidden_states.dtype)
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = torch.from_numpy(mask_feature_indices).to(hidden_states.device)
                hidden_states[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        #print(encoder_outputs.shape)
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def extract_wav2vec2(file_folder, destpath, fps, inference_length=16000*20):
    wav2vec_model = Wav2Vec2Model.from_pretrained("/home/ma-user/work/datasets/hub/transformer/wav2vec2-base-960h")
    wav2vec_model.feature_extractor._freeze_parameters()
    wav2vec_model = wav2vec_model.cuda()
    wav2vec_model.eval()
    audio_mean = np.load("/home/ma-user/work/datasets/beat_cache/beat_english_15_141/train/wave16k/npy_mean.npy")
    audio_std = np.load("/home/ma-user/work/datasets/beat_cache/beat_english_15_141/train/wave16k/npy_std.npy")
    if not os.path.exists(destpath): os.mkdir(destpath)
    with torch.no_grad():
        for file_name in tqdm(os.listdir(file_folder)):
            if "mean" in file_name or "std" in file_name or "pynb" in file_name: continue
            audio_np = np.load(file_folder+file_name)
            audio_np = (audio_np-audio_mean)/audio_std
            audio_torch = torch.from_numpy(audio_np).cuda()
            audio_torch = audio_torch.reshape(1, -1)
            #print(audio_torch.shape, audio_np.shape)
            
            if audio_torch.shape[1] > inference_length:
                num_div = audio_torch.shape[1] // inference_length
                remain = audio_torch.shape[1] % inference_length
                for i in range(num_div):
                    audio_feat = wav2vec_model(audio_torch[:, i*inference_length:(i+1)*inference_length]).last_hidden_state.cpu().numpy().reshape(-1, 768)
                    if i == 0:
                        audio_feat_all = audio_feat
                    else:
                        audio_feat_all = np.concatenate((audio_feat_all, audio_feat), 0)
                if remain > 1600: #0.25s
                    audio_feat = wav2vec_model(audio_torch[:, num_div*inference_length:num_div*inference_length+remain]).last_hidden_state.cpu().numpy().reshape(-1, 768)
                    audio_feat_all = np.concatenate((audio_feat_all, audio_feat), 0)
            else:
                audio_feat_all = wav2vec_model(audio_torch).last_hidden_state.cpu().numpy().reshape(-1, 768)
            #print(audio_feat_all.shape, audio_np.shape[0]/16000*15, torch.cuda.memory_cached() / 1E9)
            np.save(destpath+file_name, audio_feat_all)

def extract_melspec(file, destpath, fps, n_mels=128):
    fs,X = wav.read(file)
    X = X.astype(float)/math.pow(2,15)
    target_sr = 48000
    X_48k = librosa.resample(X, orig_sr=fs, target_sr=target_sr, res_type="kaiser_best")
    n_fft=int(target_sr*0.13)
    hop_len=int(target_sr/fps)
    C = librosa.feature.melspectrogram(y=X_48k, sr=target_sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels, fmin=0.0, fmax=8000)
    #C2 = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=1024, hop_length=512)
    #print(C.shape, C2.shape)
    C = np.log(C)
    np.save(destpath,np.transpose(C))

    
if __name__ == "__main__":
    #calculate mean and build cache for data. 
    target_fps = 15
    ori_data_path = f"/home/ma-user/work/datasets/beat_cache/beat_english_{target_fps}_141/"
    for data_type in ["train", "val", "test"]:
        extract_wav2vec2(ori_data_path+data_type+"/wave16k/", ori_data_path+data_type+f"/wav2vec2_{target_fps}/", target_fps)