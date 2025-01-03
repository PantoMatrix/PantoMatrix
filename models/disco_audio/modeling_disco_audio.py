"""
motion_in and motion_out are all (bs, t, c), not (bs, t, j, c//j)
input:
    audio: (bs, audio_t)
    speaker_id: (bs, 1)
    seed_frames: int
    seed_motion: (bs, t, j*6) # rot6d
output:
    motion: (bs, t, j*6) # rot6d
    motion_axis_angle: (bs, t, j*3) # axis-angle
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from .configuration_disco_audio import DiscoAudioConfig

# ------------------ utils ---------------------- #
MASK_DICT = {
    "local_upper": [
        False, False, False, True, False, False, True, False, False, True,
        False, False, True, True, True, True, True, True, True, True,
        True, True, False, False, False, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True
    ],
    "local_full": [False] + [True]*54
}

def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotation_6d_to_axis_angle(rot6d):
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot6d))

def recover_from_mask_ts(selected_motion: torch.Tensor, mask: list[bool]) -> torch.Tensor:
    device = selected_motion.device
    dtype = selected_motion.dtype
    mask_arr = torch.tensor(mask, dtype=torch.bool, device=device)
    j = len(mask_arr)
    sum_mask = mask_arr.sum().item()
    c_channels = selected_motion.shape[-1] // sum_mask
    new_shape = selected_motion.shape[:-1] + (sum_mask, c_channels)
    selected_motion = selected_motion.reshape(new_shape)
    out_shape = list(selected_motion.shape[:-2]) + [j, c_channels]
    recovered = torch.zeros(out_shape, dtype=dtype, device=device)
    recovered[..., mask_arr, :] = selected_motion
    final_shape = list(recovered.shape[:-2]) + [j * c_channels]
    recovered = recovered.reshape(final_shape)
    return recovered


# ------------------ network ---------------------- #
class BasicBlock(nn.Module):
    """Basic 1D residual block."""
    def __init__(self, inplanes, planes, ker_size, stride=1, first_dilation=None, norm_layer=nn.BatchNorm1d, act_layer=nn.LeakyReLU):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=ker_size, stride=stride, 
                               padding=first_dilation, dilation=1, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=ker_size, padding=ker_size//2, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, stride=stride, kernel_size=ker_size, padding=first_dilation, bias=True),
                norm_layer(planes)
            )

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x


class WavEncoder(nn.Module):
    """Waveform encoder that uses stacked residual blocks."""
    def __init__(self, out_dim):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            BasicBlock(1, 32, 15, 5, first_dilation=1600),
            BasicBlock(32,32,15,6,first_dilation=0),
            BasicBlock(32,32,15,1,first_dilation=7),
            BasicBlock(32,64,15,6,first_dilation=0),
            BasicBlock(64,64,15,1,first_dilation=7),
            BasicBlock(64,128,15,6,first_dilation=0),
        )

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)


class MLP(nn.Module):
    """A simple MLP for projection."""
    def __init__(self, in_dim, middle_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, out_dim)
        self.act = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Empty(nn.Module):
    """Empty module that returns input as is."""
    def forward(self, x):
        return x


class DiscoAudioPreTrainedModel(PreTrainedModel):
    config_class = DiscoAudioConfig
    base_model_prefix = "camn_audio"

    def _init_weights(self, module):
        pass


class DiscoAudioModel(DiscoAudioPreTrainedModel):
    """DiscoAudio model for audio-driven motion generation.

    This model assumes that the config (DiscoAudioConfig) can be initialized from a dict-like object
    or OmegaConf directly by passing them as kwargs. For example:
    
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/camn_audio.yaml")
        config = DiscoAudioConfig(config_obj=cfg.model)

    This way all attributes from cfg.model become config attributes without having to manually map each one.
    """
    def __init__(self, config: DiscoAudioConfig):
        super().__init__(config)
        self.pose_rep = config.pose_rep
        self.cfg = config
        self.audio_encoder = WavEncoder(self.cfg.audio_f)
        self.speaker_embedding = nn.Embedding(self.cfg.speaker_dims, self.cfg.speaker_f) if self.cfg.speaker_f > 0 else None
        self.motion_encoder = Empty()
        self.joint_mask = MASK_DICT[config.joint_mask]

        self.audio_encoder_c1 = MLP(self.cfg.audio_f, self.cfg.hidden_size, self.cfg.audio_f)
        self.audio_encoder_c2 = MLP(self.cfg.audio_f, self.cfg.hidden_size, self.cfg.audio_f)
        self.audio_encoder_r = MLP(self.cfg.audio_f, self.cfg.hidden_size, self.cfg.audio_f)

        self.selector = MLP(self.cfg.audio_f, self.cfg.hidden_size, 2)
        self.softmax = nn.Softmax(dim=2)

        input_dim_body = self.cfg.pose_dims+1+self.cfg.speaker_f+self.cfg.audio_f*2
        self.body_motion_decoder = nn.LSTM(
            input_dim_body, hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.n_layer, batch_first=True,
            bidirectional=True, dropout=self.cfg.dropout_prob
        )
        self.body_out = MLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.pose_dims)


    def forward(self, audio, speaker_id, seed_frames=4, seed_motion=None, return_axis_angle=True):
        audio_feat = self.audio_encoder(audio)
        bs, t, _ = audio_feat.shape

        if self.speaker_embedding is not None:
            speaker_feat = self.speaker_embedding(speaker_id)
            speaker_feat = speaker_feat.repeat(1, t, 1)
        else:
            speaker_feat = torch.zeros(bs, t, 0, device=audio.device)

        if seed_motion is None:
            seed_motion = torch.zeros(bs, t, self.cfg.pose_dims+1, device=audio.device)
            seed_motion[:, :seed_frames, -1] = 1
        else:
            _, t_m, _ = seed_motion.shape
            seed_motion_pad = torch.zeros(bs, t_m, self.cfg.pose_dims+1, device=audio.device)
            seed_motion_pad[:, :seed_frames, :-1] = seed_motion[:, :seed_frames]
            seed_motion_pad[:, :seed_frames, -1] = 1
            seed_motion = seed_motion_pad
            if t_m != t:
                diff_length = t_m - t
                if diff_length > 0:
                    seed_motion = seed_motion[:, :t, :]
                else:
                    seed_motion = torch.cat((seed_motion, seed_motion[:, -diff_length:, :]), 1)
        
        audio_feat_c1 = self.audio_encoder_c1(audio_feat)
        audio_feat_c2 = self.audio_encoder_c2(audio_feat)
        audio_feat_r = self.audio_encoder_r(audio_feat)
        weight_c = self.softmax(self.selector(audio_feat))
        audio_feat_c = weight_c[:, :, 0:1] * audio_feat_c1 + weight_c[:, :, 1:2] * audio_feat_c2 
        audio_feat = torch.cat((audio_feat_c, audio_feat_r), dim=2)

        in_fea = torch.cat((audio_feat, speaker_feat, seed_motion), dim=2)
        body_out, _ = self.body_motion_decoder(in_fea)
        body_out = body_out[:, :, :self.cfg.hidden_size] + body_out[:, :, self.cfg.hidden_size:]
        recombine = self.body_out(body_out)

        motion_axis_angle = None
        if return_axis_angle:
            motion_axis_angle = rotation_6d_to_axis_angle(recombine.reshape(-1, self.cfg.pose_dims//6, 6)).reshape(bs, t, -1)
            motion_axis_angle = recover_from_mask_ts(motion_axis_angle, self.joint_mask)
        return {
            "motion": recombine,
            "motion_axis_angle": motion_axis_angle,
            "audio_fea_c": audio_feat_c,   
            "audio_fea_r": audio_feat_r,
            }