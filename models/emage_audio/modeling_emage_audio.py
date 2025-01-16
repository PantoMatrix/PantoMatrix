import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers import PreTrainedModel
from .configuration_emage_audio import EmageAudioConfig, EmageVQVAEConvConfig, EmageVAEConvConfig
from .processing_emage_audio import Quantizer, VQEncoderV5, VQDecoderV5, WavEncoder, MLP, PeriodicPositionalEncoding, VQEncoderV6, recover_from_mask_ts, rotation_6d_to_axis_angle, velocity2position, axis_angle_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_rotation_6d


def inverse_selection_tensor(filtered_t, selection_array, n):
    selection_array = torch.from_numpy(selection_array).cuda()
    original_shape_t = torch.zeros((n, 165)).cuda()
    selected_indices = torch.where(selection_array == 1)[0]
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    return original_shape_t

class EmageVAEConv(PreTrainedModel):
    config_class = EmageVAEConvConfig
    base_model_prefix = "emage_vaeconv"
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VQEncoderV5(config)
        self.decoder = VQDecoderV5(config)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        rec_pose = self.decoder(pre_latent)
        return {
            "rec_pose": rec_pose
            }

class EmageVQVAEConv(PreTrainedModel):
    config_class = EmageVQVAEConvConfig
    base_model_prefix = "emage_vqvaeconv"
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VQEncoderV5(config)
        self.quantizer = Quantizer(config.vae_codebook_size, config.vae_length, config.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(config)
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {"poses_feat":vq_latent,"embedding_loss":embedding_loss,"perplexity":perplexity,"rec_pose": rec_pose}
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose
    def decode_from_latent(self, latent):
        # print(latent.shape)
        z_flattened = latent.contiguous().view(-1, self.quantizer.e_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(self.quantizer.embedding.weight**2, dim=1) - 2*torch.matmul(z_flattened, self.quantizer.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        # print(min_encoding_indices.shape)
        indices = min_encoding_indices.view(latent.shape[0], latent.shape[1])
        z_q = self.quantizer.get_codebook_entry(indices)
        rec_pose = self.decoder(z_q)
        return rec_pose

class EmageVQModel(nn.Module):
    def __init__(self, face_model, upper_model, hands_model, lower_model, global_model):
        super().__init__()
        self.joint_mask_upper = [
          False, False, False, True, False, False, True, False, False, True,
          False, False, True, True, True, True, True, True, True, True,
          True, True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False
        ]
        self.joint_mask_lower = [
          True, True, True, False, True, True, False, True, True, False,
          True, True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False
        ]
        self.vq_model_face = face_model
        self.vq_model_upper = upper_model
        self.vq_model_hands = hands_model
        self.vq_model_lower = lower_model
        self.global_motion = global_model

    def spilt_inputs(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        bs, t, j6 = smplx_body_rot6d.shape
        smplx_body_rot6d = smplx_body_rot6d.reshape(bs, t, j6//6, 6)
        jaw_rot6d = smplx_body_rot6d[:, :, 22:23, :].reshape(bs, t, 6)
        face = torch.cat([jaw_rot6d, expression], dim=2)
        upper_rot6d = smplx_body_rot6d[:, :,self.joint_mask_upper, :].reshape(bs, t, 78)
        hands_rot6d = smplx_body_rot6d[:, :,25:55, :].reshape(bs, t, 180)
        lower_rot6d = smplx_body_rot6d[:, :,self.joint_mask_lower, :].reshape(bs, t, 54)
        tar_contact = torch.zeros(bs, t, 4, device=smplx_body_rot6d.device) if tar_contact is None else tar_contact
        tar_trans = torch.zeros(bs, t, 3, device=smplx_body_rot6d.device) if tar_trans is None else tar_trans
        lower = torch.cat([lower_rot6d, tar_trans, tar_contact], dim=2)
        return dict(face=face, upper=upper_rot6d, hands=hands_rot6d, lower=lower)
    
    def map2index(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        inputs = self.spilt_inputs(smplx_body_rot6d, expression, tar_contact=tar_contact, tar_trans=tar_trans)
        face_index = self.vq_model_face.map2index(inputs["face"])
        upper_index = self.vq_model_upper.map2index(inputs["upper"])
        hands_index = self.vq_model_hands.map2index(inputs["hands"])
        lower_index = self.vq_model_lower.map2index(inputs["lower"])
        return dict(face=face_index, upper=upper_index, hands=hands_index, lower=lower_index)
    
    def map2latent(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        inputs = self.spilt_inputs(smplx_body_rot6d, expression,tar_contact=tar_contact, tar_trans=tar_trans)
        face_latent = self.vq_model_face.map2latent(inputs["face"])
        upper_latent = self.vq_model_upper.map2latent(inputs["upper"])
        hands_latent = self.vq_model_hands.map2latent(inputs["hands"])
        lower_latent = self.vq_model_lower.map2latent(inputs["lower"])
        return dict(face=face_latent, upper=upper_latent, hands=hands_latent, lower=lower_latent)
    
    def decode(self, face_index=None, upper_index=None, hands_index=None, lower_index=None, 
               face_latent=None, upper_latent=None, hands_latent=None, lower_latent=None, 
            get_global_motion=False, ref_trans=None):
        
        for input_tensor in [face_index, upper_index, hands_index, lower_index, face_latent, upper_latent, hands_latent, lower_latent]:
            if input_tensor is not None:
                bs, t = input_tensor.shape[:2]
                break
  
        if face_index is not None:
            face_mix = self.vq_model_face.decode(face_index) # bs, t, 106
            face_jaw_6d, expression = face_mix[:, :, :6], face_mix[:, :, 6:]
            face_jaw = rotation_6d_to_axis_angle(face_jaw_6d)
        elif face_latent is not None:
            face_mix = self.vq_model_face.decode_from_latent(face_latent)
            face_jaw_6d, expression = face_mix[:, :, :6], face_mix[:, :, 6:]
            face_jaw = rotation_6d_to_axis_angle(face_jaw_6d)
        else:
            face_jaw = torch.zeros(bs, t, 3, device=self.vq_model_face.device)
            expression = torch.zeros(bs, t, 100, device=self.vq_model_face.device)

        if upper_index is not None:
            # print(upper_index)
            upper_6d = self.vq_model_upper.decode(upper_index) # bs, t, 78
            upper = rotation_6d_to_axis_angle(upper_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif upper_latent is not None:
            upper_6d = self.vq_model_upper.decode_from_latent(upper_latent)
            upper = rotation_6d_to_axis_angle(upper_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            upper = torch.zeros(bs, t, 39, device=self.vq_model_upper.device)

        if hands_index is not None:
            hands_6d = self.vq_model_hands.decode(hands_index)
            hands = rotation_6d_to_axis_angle(hands_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif hands_latent is not None:
            hands_6d = self.vq_model_hands.decode_from_latent(hands_latent)
            hands = rotation_6d_to_axis_angle(hands_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            hands = torch.zeros(bs, t, 90, device=self.vq_model_hands.device)
        
        if lower_index is not None:
            lower_mix = self.vq_model_lower.decode(lower_index)
            lower_6d, transfoot = lower_mix[:, :, :-7], lower_mix[:, :, -7:]
            lower = rotation_6d_to_axis_angle(lower_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif lower_latent is not None:
            lower_mix = self.vq_model_lower.decode_from_latent(lower_latent)
            lower_6d, transfoot = lower_mix[:, :, :-7], lower_mix[:, :, -7:]
            lower = rotation_6d_to_axis_angle(lower_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            lower = torch.zeros(bs, t, 27, device=self.vq_model_lower.device)
            transfoot = torch.zeros(bs, t, 7, device=self.vq_model_lower.device)
            lower_6d = axis_angle_to_rotation_6d(lower.reshape(bs, t, -1, 3)).reshape(bs, t, -1)
            lower_mix = torch.cat([lower_6d, transfoot], dim=-1)

        upper2all = recover_from_mask_ts(upper, self.joint_mask_upper)
        hands2all = recover_from_mask_ts(hands, [False]*25+[True]*30)
        lower2all = recover_from_mask_ts(lower, self.joint_mask_lower)
        
        all_motion_axis_angle = upper2all + hands2all + lower2all
        all_motion_axis_angle[:, :, 22*3:22*3+3] = face_jaw
        all_motion_rot6d = axis_angle_to_rotation_6d(all_motion_axis_angle.reshape(bs, t, 55, 3)).reshape(bs, t, 55*6)

        all_motion4inference = torch.cat([all_motion_rot6d, transfoot], dim=2) # 330 + 3 + 4
        
        global_motion = None
        if get_global_motion:
            global_motion = self.get_global_motion(lower_mix, ref_trans)
        return dict(expression=expression, all_motion4inference=all_motion4inference, motion_axis_angle=all_motion_axis_angle, trans=global_motion)
    
    def get_global_motion(self, lower_body, ref_trans):
        global_motion = self.global_motion(lower_body)
        rec_trans_v_s = global_motion["rec_pose"][:, :, 54:57]
        if len(ref_trans.shape) == 2:
            ref_trans = ref_trans.unsqueeze(0).repeat(rec_trans_v_s.shape[0], 1, 1)
        
        rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1], 1/30, ref_trans[:, 0, 0:1])
        rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3], 1/30, ref_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        global_motion = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        return global_motion


class EmageAudioModel(PreTrainedModel):
    config_class = EmageAudioConfig
    base_model_prefix = "emage_audio"
    def __init__(self, config: EmageAudioConfig):
        super().__init__(config)
        self.cfg = config
        # audio encoder
        self.audio_encoder_face = WavEncoder(self.cfg.audio_f)
        self.audio_encoder_body = WavEncoder(self.cfg.audio_f)
        #speaker id
        self.speaker_embedding_body = nn.Embedding(self.cfg.speaker_dims, self.cfg.hidden_size)
        self.speaker_embedding_face = nn.Embedding(self.cfg.speaker_dims, self.cfg.hidden_size)
        # mask embedding
        self.mask_embedding = nn.Parameter(torch.zeros(1,1,self.cfg.pose_dims+3+4))
        nn.init.normal_(self.mask_embedding, 0, self.cfg.hidden_size**-0.5)
        # nn.init.normal_(self.speaker_embedding_body.weight, 0, self.cfg.hidden_size/2**-0.5)
        # nn.init.normal_(self.speaker_embedding_face.weight, 0, self.cfg.hidden_size*2**-0.5)

        # motion pre encoder
        args_top = copy.deepcopy(self.cfg)
        args_top.vae_layer = 3
        args_top.vae_length = self.cfg.motion_f
        args_top.vae_test_dim = self.cfg.pose_dims+3+4
        self.motion_encoder = VQEncoderV6(args_top)
        self.bodyhints_face = MLP(self.cfg.motion_f, self.cfg.hidden_size, self.cfg.motion_f)
        self.bodyhints_body = MLP(self.cfg.motion_f, self.cfg.hidden_size, self.cfg.motion_f)
        # motion encoder
        self.audio_body_motion_proj = nn.Linear(self.cfg.audio_f, self.cfg.hidden_size)
        self.moton_proj = nn.Linear(self.cfg.motion_f, self.cfg.hidden_size)
        self.position_embeddings = PeriodicPositionalEncoding(self.cfg.hidden_size, period=self.cfg.pose_length, max_seq_len=self.cfg.pose_length)
        self.transformer_en_layer = nn.TransformerEncoderLayer(d_model=self.cfg.hidden_size,nhead=4,dim_feedforward=self.cfg.hidden_size*2)
        self.motion_self_encoder = nn.TransformerEncoder(self.transformer_en_layer, num_layers=1)
        # coss attn
        self.audio_motion_cross_attn_layer = nn.TransformerDecoderLayer(d_model=self.cfg.hidden_size,nhead=4,dim_feedforward=self.cfg.hidden_size*2)
        self.audio_motion_cross_attn = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=8)
        # feed forward
        self.motion2latent_upper = MLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.hidden_size)
        self.motion2latent_hands = MLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.hidden_size)
        self.motion2latent_lower = MLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.hidden_size)
        # refine
        self.body_motion_decoder_upper = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=1)
        self.body_motion_decoder_hands = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=1)
        self.body_motion_decoder_lower = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=1)
        # deocder
        self.motion_out_proj_upper = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_out_proj_hands = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_out_proj_lower = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_cls_upper = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_cls_hands = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_cls_lower = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)

        # face decoder
        self.audio_face_motion_proj = nn.Linear(self.cfg.audio_f+self.cfg.motion_f, self.cfg.hidden_size)
        self.face_motion_decoder = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=4)
        self.face_out_proj = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.face_cls = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)
    
    def forward(self, audio, speaker_id, masked_motion, mask, use_audio=True):
        # mask motion
        masked_embeddings = self.mask_embedding.expand_as(masked_motion)
        masked_motion = torch.where(mask==1, masked_embeddings, masked_motion)

        # motion token (spatial hints)
        body_hint = self.motion_encoder(masked_motion)
        body_hint_body = self.bodyhints_body(body_hint)
        body_hint_face = self.bodyhints_face(body_hint)
        
        audio2face_fea = self.audio_encoder_face(audio)
        audio2body_fea = self.audio_encoder_body(audio)

        if audio2face_fea.shape[1] > body_hint_face.shape[1]:
            audio2face_fea = audio2face_fea[:, :body_hint_face.shape[1]]
        if audio2body_fea.shape[1] > body_hint_face.shape[1]:
            audio2face_fea = audio2face_fea[:, :body_hint_face.shape[1]]

        bs, t, _ = audio2face_fea.shape

        speaker_motion_fea_proj = self.speaker_embedding_body(speaker_id).repeat(1, t, 1)
        speaker_face_fea_proj = self.speaker_embedding_face(speaker_id).repeat(1, t, 1)
        
        audio2face_fea_proj = self.audio_face_motion_proj(torch.cat([audio2face_fea, body_hint_face], dim=2))
        # audio2face_fea_proj = self.position_embeddings(audio2face_fea_proj)
        # audio2face_fea_proj = speaker_face_fea_proj + audio2face_fea_proj
        face_proj = self.position_embeddings(speaker_face_fea_proj)
        decode_face = self.face_motion_decoder(tgt=face_proj.permute(1,0,2), memory=audio2face_fea_proj.permute(1,0,2)).permute(1,0,2)
        face_latent = self.face_out_proj(decode_face)
        classify_face = self.face_cls(face_latent)

        # motion self attn (temporal)
        masked_motion_proj = self.moton_proj(body_hint_body)
        masked_motion_proj = self.position_embeddings(masked_motion_proj)
        masked_motion_proj = speaker_motion_fea_proj + masked_motion_proj
        motion_fea = self.motion_self_encoder(masked_motion_proj.permute(1,0,2)).permute(1,0,2)

        # audio_cross_attn
        
        audio2body_fea_proj = self.audio_body_motion_proj(audio2body_fea)
        # audio2body_fea_proj = self.position_embeddings(audio2body_fea_proj)
        # audio2body_fea_proj = speaker_motion_fea_proj + audio2body_fea_proj
        motion_fea = motion_fea + speaker_motion_fea_proj
        motion_fea = self.position_embeddings(motion_fea)
        audio2body_fea_cross = self.audio_motion_cross_attn(tgt=motion_fea.permute(1,0,2), memory=audio2body_fea_proj.permute(1,0,2)).permute(1,0,2)
        if not use_audio:
          audio2body_fea_cross = audio2body_fea_cross * 0.
        motion_fea = motion_fea + audio2body_fea_cross 

        # mlp
        upper_latent = self.motion2latent_upper(motion_fea)
        hands_latent = self.motion2latent_hands(motion_fea)
        lower_latent = self.motion2latent_lower(motion_fea)

        # refine
        motion_upper_refine = self.body_motion_decoder_upper(tgt=upper_latent.permute(1,0,2)+speaker_motion_fea_proj.permute(1,0,2), memory=(hands_latent+lower_latent).permute(1,0,2)).permute(1,0,2)
        motion_hands_refine = self.body_motion_decoder_hands(tgt=hands_latent.permute(1,0,2)+speaker_motion_fea_proj.permute(1,0,2), memory=(upper_latent+lower_latent).permute(1,0,2)).permute(1,0,2)
        motion_lower_refine = self.body_motion_decoder_lower(tgt=lower_latent.permute(1,0,2)+speaker_motion_fea_proj.permute(1,0,2), memory=(upper_latent+hands_latent).permute(1,0,2)).permute(1,0,2)
        upper_latent = self.motion_out_proj_upper(upper_latent + motion_upper_refine)
        hands_latent = self.motion_out_proj_hands(hands_latent + motion_hands_refine)
        lower_latent = self.motion_out_proj_lower(lower_latent + motion_lower_refine)

        # decode body
        classify_upper = self.motion_cls_upper(upper_latent)
        classify_hands = self.motion_cls_hands(hands_latent)
        classify_lower = self.motion_cls_lower(lower_latent)

        return {
            "rec_face": face_latent,
            "rec_upper": upper_latent,
            "rec_hands": hands_latent,
            "rec_lower": lower_latent,
            "cls_face": classify_face,
            "cls_upper": classify_upper,
            "cls_hands": classify_hands,
            "cls_lower": classify_lower,
        }
    
    def inference(self, audio, speaker_id, vq_model, masked_motion=None, mask=None):
        # generate default mask and masked motion if not provided
        length = audio.shape[1] * 30 // 16000
        bs = audio.shape[0]

        fake_axis_angle = torch.zeros(bs, length, 55, 3).to(audio.device)
        fake_motion = axis_angle_to_rotation_6d(fake_axis_angle).reshape(bs, length, -1)
        fake_foot_and_trans = torch.zeros(bs, length, 7).to(audio.device)
        fake_motion = torch.cat([fake_motion, fake_foot_and_trans], dim=-1) 
        if masked_motion is not None:
            fake_motion[:, :masked_motion.shape[1]] = masked_motion 
        masked_motion = fake_motion

        fake_mask = torch.ones_like(masked_motion)
        if mask is not None:
            fake_mask[:, :mask.shape[1]] = mask 
        mask = fake_mask

        # print(length, masked_motion.shape, mask.shape)
        
        # Autoregressive inference
        bs, total_len, c = masked_motion.shape
        window = self.cfg.pose_length
        pre_frames = self.cfg.seed_frames
        rounds = (total_len - pre_frames) // (window - pre_frames)
        remain = (total_len - pre_frames) % (window - pre_frames)
        
        rec_all_face = []
        rec_all_lower = []
        rec_all_upper = []
        rec_all_hands = []
        cls_all_face = []
        cls_all_lower = []
        cls_all_upper = []
        cls_all_hands = []
        
        last_motion = masked_motion[:, :pre_frames, :]
        for i in range(rounds):
            start_idx = i*(window - pre_frames)
            end_idx = start_idx + window

            window_mask = mask[:, start_idx:end_idx, :].clone()
            window_motion = masked_motion[:, start_idx:end_idx, :].clone()
            window_motion[:, :pre_frames, :] = torch.where(
                (window_mask[:, :pre_frames, :] == 0),
                masked_motion[:, start_idx:start_idx+pre_frames, :],
                last_motion,
            )
            window_mask[:, :pre_frames, :] = 0

            audio_slice_len = (end_idx - start_idx)*(16000//30)
            audio_slice = audio[:, start_idx*(16000//30) : start_idx*(16000//30)+audio_slice_len]
            # print(i, audio_slice.shape, speaker_id.shape, window_motion.shape, window_mask.shape)
            net_out_val = self.forward(audio_slice, speaker_id, masked_motion=window_motion, mask=window_mask, use_audio=True)
       
            _, cls_face =  torch.max(F.log_softmax(net_out_val["cls_face"], dim=2), dim=2)
            _, cls_upper =  torch.max(F.log_softmax(net_out_val["cls_upper"], dim=2), dim=2)
            _, cls_hands =  torch.max(F.log_softmax(net_out_val["cls_hands"], dim=2), dim=2)
            _, cls_lower =  torch.max(F.log_softmax(net_out_val["cls_lower"], dim=2), dim=2)

            face_latent = net_out_val["rec_face"] if self.cfg.lf > 0 and self.cfg.cf == 0 else None
            upper_latent = net_out_val["rec_upper"] if self.cfg.lu > 0 and self.cfg.cu == 0 else None
            hands_latent = net_out_val["rec_hands"] if self.cfg.lh > 0 and self.cfg.ch == 0 else None
            lower_latent = net_out_val["rec_lower"] if self.cfg.ll > 0 and self.cfg.cl == 0 else None
            face_index = cls_face if self.cfg.cf > 0 else None
            upper_index = cls_upper if self.cfg.cu > 0 else None
            hands_index = cls_hands if self.cfg.ch > 0 else None
            lower_index = cls_lower if self.cfg.cl > 0 else None

            decode_dict = vq_model.decode(
            face_latent=face_latent, upper_latent=upper_latent, lower_latent=lower_latent, hands_latent=hands_latent,
            face_index=face_index, upper_index=upper_index, lower_index=lower_index, hands_index=hands_index,)
            
            # decode_dict = vq_model.decode(face_latent=net_out_val["rec_face"], upper_index=net_out_val["cls_upper"], hands_index=net_out_val["cls_hands"], lower_index=net_out_val["cls_lower"])
            
            last_motion = decode_dict["all_motion4inference"][:, -pre_frames:, :]
            rec_all_face.append(net_out_val["rec_face"][:, :-pre_frames, :])
            rec_all_upper.append(net_out_val["rec_upper"][:, :-pre_frames, :])
            rec_all_hands.append(net_out_val["rec_hands"][:, :-pre_frames, :])
            rec_all_lower.append(net_out_val["rec_lower"][:, :-pre_frames, :])
            cls_all_face.append(net_out_val["cls_face"][:, :-pre_frames])
            cls_all_upper.append(net_out_val["cls_upper"][:, :-pre_frames])
            cls_all_hands.append(net_out_val["cls_hands"][:, :-pre_frames])
            cls_all_lower.append(net_out_val["cls_lower"][:, :-pre_frames])

        if remain > pre_frames:
            final_start = rounds*(window - pre_frames)
            final_end = final_start + pre_frames + remain

            final_mask = mask[:, final_start:final_end, :].clone()
            final_motion = masked_motion[:, final_start:final_end, :].clone()
            final_motion[:, :pre_frames, :] = torch.where(
                (final_mask[:, :pre_frames, :] == 0),
                masked_motion[:, final_start:final_start+pre_frames, :],
                last_motion,
            )
            final_mask[:, :pre_frames, :] = 0

            audio_slice_len = (final_end - final_start)*(16000//30)
            audio_slice = audio[:, final_start*(16000//30) : final_start*(16000//30)+audio_slice_len]
            net_out_val = self.forward(audio_slice, speaker_id, masked_motion=final_motion, mask=final_mask, use_audio=True)

            _, cls_face =  torch.max(F.log_softmax(net_out_val["cls_face"], dim=2), dim=2)
            _, cls_upper =  torch.max(F.log_softmax(net_out_val["cls_upper"], dim=2), dim=2)
            _, cls_hands =  torch.max(F.log_softmax(net_out_val["cls_hands"], dim=2), dim=2)
            _, cls_lower =  torch.max(F.log_softmax(net_out_val["cls_lower"], dim=2), dim=2)

            face_latent = net_out_val["rec_face"] if self.cfg.lf > 0 and self.cfg.cf == 0 else None
            upper_latent = net_out_val["rec_upper"] if self.cfg.lu > 0 and self.cfg.cu == 0 else None
            hands_latent = net_out_val["rec_hands"] if self.cfg.lh > 0 and self.cfg.ch == 0 else None
            lower_latent = net_out_val["rec_lower"] if self.cfg.ll > 0 and self.cfg.cl == 0 else None
            face_index = cls_face if self.cfg.cf > 0 else None
            upper_index = cls_upper if self.cfg.cu > 0 else None
            hands_index = cls_hands if self.cfg.ch > 0 else None
            lower_index = cls_lower if self.cfg.cl > 0 else None

            decode_dict = vq_model.decode(
            face_latent=face_latent, upper_latent=upper_latent, lower_latent=lower_latent, hands_latent=hands_latent,
            face_index=face_index, upper_index=upper_index, lower_index=lower_index, hands_index=hands_index,)

            rec_all_face.append(net_out_val["rec_face"])
            rec_all_upper.append(net_out_val["rec_upper"])
            rec_all_hands.append(net_out_val["rec_hands"])
            rec_all_lower.append(net_out_val["rec_lower"])
            cls_all_face.append(net_out_val["cls_face"])
            cls_all_upper.append(net_out_val["cls_upper"])
            cls_all_hands.append(net_out_val["cls_hands"])
            cls_all_lower.append(net_out_val["cls_lower"])

        rec_all_face = torch.cat(rec_all_face, dim=1) 
        rec_all_upper = torch.cat(rec_all_upper, dim=1) 
        rec_all_hands = torch.cat(rec_all_hands, dim=1) 
        rec_all_lower = torch.cat(rec_all_lower, dim=1) 
        cls_all_face = torch.cat(cls_all_face, dim=1)
        cls_all_upper = torch.cat(cls_all_upper, dim=1) 
        cls_all_hands = torch.cat(cls_all_hands, dim=1) 
        cls_all_lower = torch.cat(cls_all_lower, dim=1) 

        return {
            "rec_face": rec_all_face,
            "rec_upper": rec_all_upper,
            "rec_hands": rec_all_hands,
            "rec_lower": rec_all_lower,
            "cls_face": cls_all_face,
            "cls_upper": cls_all_upper,
            "cls_hands": cls_all_hands,
            "cls_lower": cls_all_lower,
        }










