import copy
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from .utils.layer import BasicBlock
from .motion_encoder import * 


class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        # print(wav_data.shape)   
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)

    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, out_dim)
        )
    def forward(self, inputs):
        out = self.mlp(inputs)
        return out


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=60): 
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1) # (1, repeat_num, period, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # print(self.pe.shape, x.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class MAGE_Transformer(nn.Module):
    def __init__(self, args):
        super(MAGE_Transformer, self).__init__()
        self.args = args   
        # with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
        #     self.lang_model = pickle.load(f)
        #     pre_trained_embedding = self.lang_model.word_embedding_weights
        # self.text_pre_encoder_face = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=args.t_fix_pre)
        # self.text_encoder_face = nn.Linear(300, args.audio_f) 
        # self.text_encoder_face = nn.Linear(300, args.audio_f) 
        # self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=args.t_fix_pre)
        # self.text_encoder_body = nn.Linear(300, args.audio_f) 
        # self.text_encoder_body = nn.Linear(300, args.audio_f) 

        self.audio_pre_encoder_face = WavEncoder(args.audio_f, audio_in=1)
        self.audio_pre_encoder_body = WavEncoder(args.audio_f, audio_in=1)
        
        # self.at_attn_face = nn.Linear(args.audio_f*2, args.audio_f*2)
        # self.at_attn_body = nn.Linear(args.audio_f*2, args.audio_f*2)
        
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = args.motion_f
        args_top.vae_test_dim = args.pose_dims+3+4
        self.motion_encoder = VQEncoderV6(args_top) # masked motion to latent bs t 333 to bs t 256
        
        # face decoder 
        self.feature2face = nn.Linear(args.audio_f*2, args.hidden_size)
        self.face2latent = nn.Linear(args.hidden_size, args.vae_codebook_size)
        self.transformer_de_layer = nn.TransformerDecoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.face_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=4)
        self.position_embeddings = PeriodicPositionalEncoding(self.args.hidden_size, period=self.args.pose_length, max_seq_len=self.args.pose_length)
        
        # motion decoder
        self.transformer_en_layer = nn.TransformerEncoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.motion_self_encoder = nn.TransformerEncoder(self.transformer_en_layer, num_layers=1)
        self.audio_feature2motion = nn.Linear(args.audio_f, args.hidden_size)
        self.feature2motion = nn.Linear(args.motion_f, args.hidden_size)

        self.bodyhints_face = MLP(args.motion_f, args.hidden_size, args.motion_f)
        self.bodyhints_body = MLP(args.motion_f, args.hidden_size, args.motion_f)
        self.motion2latent_upper = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_hands = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_lower = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.wordhints_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=8)
        
        self.upper_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.hands_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.lower_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)

        self.face_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.upper_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.hands_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.lower_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)

        self.mask_embeddings = nn.Parameter(torch.zeros(1, 1, self.args.pose_dims+3+4))
        self.motion_down_upper = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_upper = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self._reset_parameters()

        self.spearker_encoder_body = nn.Embedding(25, args.hidden_size)
        self.spearker_encoder_face = nn.Embedding(25, args.hidden_size)

    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.args.hidden_size ** -0.5)
    
    def forward(self, in_audio=None, in_word=None, mask=None, is_test=None, in_motion=None, use_attentions=True, use_word=True, in_id = None):
        # in_word_face = self.text_pre_encoder_face(in_word)
        # in_word_face = self.text_encoder_face(in_word_face)
        # in_word_body = self.text_pre_encoder_body(in_word)
        # in_word_body = self.text_encoder_body(in_word_body)
        # bs, t, c = in_word_face.shape
        in_audio_face = self.audio_pre_encoder_face(in_audio)
        in_audio_body = self.audio_pre_encoder_body(in_audio)
        bs, t, c = in_audio_body.shape
        # if in_audio_face.shape[1] != in_motion.shape[1]:
        #     diff_length = in_motion.shape[1]- in_audio_face.shape[1]
        #     if diff_length < 0:
        #         in_audio_face = in_audio_face[:, :diff_length, :]
        #         in_audio_body = in_audio_body[:, :diff_length, :]
        #     else:
        #         in_audio_face = torch.cat((in_audio_face, in_audio_face[:,-diff_length:]),1)
        #         in_audio_body = torch.cat((in_audio_body, in_audio_body[:,-diff_length:]),1)

        # if use_attentions:           
        #     alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c*2)
        #     alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2)
        #     alpha_at_face = alpha_at_face.softmax(dim=-1)
        #     fusion_face = in_word_face * alpha_at_face[:,:,:,1] + in_audio_face * alpha_at_face[:,:,:,0]
        #     alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c*2)
        #     alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
        #     alpha_at_body = alpha_at_body.softmax(dim=-1)
        #     fusion_body = in_word_body * alpha_at_body[:,:,:,1] + in_audio_body * alpha_at_body[:,:,:,0]
        # else:
        fusion_face = in_audio_face
        fusion_body = in_audio_body
        
        masked_embeddings = self.mask_embeddings.expand_as(in_motion)
        masked_motion = torch.where(mask == 1, masked_embeddings, in_motion) # bs, t, 256 
        body_hint = self.motion_encoder(masked_motion) # bs t 256
        speaker_embedding_face = self.spearker_encoder_face(in_id).squeeze(2)
        speaker_embedding_body = self.spearker_encoder_body(in_id).squeeze(2)

        # decode face
        use_body_hints = True
        if use_body_hints:
            body_hint_face = self.bodyhints_face(body_hint)
            fusion_face = torch.cat([fusion_face, body_hint_face], dim=2)
        a2g_face = self.feature2face(fusion_face)
        face_embeddings = speaker_embedding_face
        face_embeddings = self.position_embeddings(face_embeddings)
        decoded_face = self.face_decoder(tgt=face_embeddings, memory=a2g_face)
        face_latent = self.face2latent(decoded_face)
        cls_face = self.face_classifier(face_latent)

        # motion spatial encoder
        body_hint_body = self.bodyhints_body(body_hint)
        motion_embeddings = self.feature2motion(body_hint_body)
        motion_embeddings = speaker_embedding_body + motion_embeddings
        motion_embeddings = self.position_embeddings(motion_embeddings)

        # bi-directional self-attention
        motion_refined_embeddings = self.motion_self_encoder(motion_embeddings) 
        
        # audio to gesture cross-modal attention
        if use_word:
            a2g_motion = self.audio_feature2motion(fusion_body)
            motion_refined_embeddings_in = motion_refined_embeddings + speaker_embedding_body
            motion_refined_embeddings_in = self.position_embeddings(motion_refined_embeddings)
            word_hints = self.wordhints_decoder(tgt=motion_refined_embeddings_in, memory=a2g_motion)
            motion_refined_embeddings = motion_refined_embeddings + word_hints
        
        # feedforward
        upper_latent = self.motion2latent_upper(motion_refined_embeddings)
        hands_latent = self.motion2latent_hands(motion_refined_embeddings)
        lower_latent = self.motion2latent_lower(motion_refined_embeddings)

        upper_latent_in = upper_latent + speaker_embedding_body
        upper_latent_in = self.position_embeddings(upper_latent_in)
        hands_latent_in = hands_latent + speaker_embedding_body
        hands_latent_in = self.position_embeddings(hands_latent_in)
        lower_latent_in = lower_latent + speaker_embedding_body
        lower_latent_in = self.position_embeddings(lower_latent_in)

        # transformer decoder
        motion_upper = self.upper_decoder(tgt=upper_latent_in, memory=hands_latent+lower_latent)
        motion_hands = self.hands_decoder(tgt=hands_latent_in, memory=upper_latent+lower_latent)
        motion_lower = self.lower_decoder(tgt=lower_latent_in, memory=upper_latent+hands_latent)
        upper_latent = self.motion_down_upper(motion_upper+upper_latent)
        hands_latent = self.motion_down_hands(motion_hands+hands_latent)
        lower_latent = self.motion_down_lower(motion_lower+lower_latent)
        cls_lower = self.lower_classifier(lower_latent)
        cls_upper = self.upper_classifier(upper_latent)
        cls_hands = self.hands_classifier(hands_latent)

        return {
            "rec_face":face_latent,
            "rec_upper":upper_latent,
            "rec_lower":lower_latent,
            "rec_hands":hands_latent,
            "cls_face":cls_face,
            "cls_upper":cls_upper,
            "cls_lower":cls_lower,
            "cls_hands":cls_hands,
            }