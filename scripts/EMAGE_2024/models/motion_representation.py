import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import copy
from .motion_encoder import * 

# ----------- AE, VAE ------------- #
class VAEConvZero(nn.Module):
    def __init__(self, args):
        super(VAEConvZero, self).__init__()
        self.encoder = VQEncoderV5(args)
        # self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        # embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(pre_latent)
        return {
            # "poses_feat":vq_latent,
            # "embedding_loss":embedding_loss,
            # "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
class VAEConv(nn.Module):
    def __init__(self, args):
        super(VAEConv, self).__init__()
        self.encoder = VQEncoderV3(args)
        self.decoder = VQDecoderV3(args)
        self.fc_mu = nn.Linear(args.vae_length, args.vae_length)
        self.fc_logvar = nn.Linear(args.vae_length, args.vae_length)
        self.variational = args.variational
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        mu, logvar = None, None
        if self.variational:
            mu = self.fc_mu(pre_latent)
            logvar = self.fc_logvar(pre_latent)
            pre_latent = reparameterize(mu, logvar)
        rec_pose = self.decoder(pre_latent)
        return {
            "poses_feat":pre_latent,
            "rec_pose": rec_pose,
            "pose_mu": mu,
            "pose_logvar": logvar,
            }
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        if self.variational:
            mu = self.fc_mu(pre_latent)
            logvar = self.fc_logvar(pre_latent)
            pre_latent = reparameterize(mu, logvar)
        return pre_latent
    
    def decode(self, pre_latent):
        rec_pose = self.decoder(pre_latent)
        return rec_pose

class VAESKConv(VAEConv):
    def __init__(self, args):
        super(VAESKConv, self).__init__(args)
        smpl_fname = args.data_path_1+'smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'
        smpl_data = np.load(smpl_fname, encoding='latin1')
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)
        self.encoder = LocalEncoder(args, edges)
        self.decoder = VQDecoderV3(args)
        
class VAEConvMLP(VAEConv):
    def __init__(self, args):
        super(VAEConvMLP, self).__init__(args)
        self.encoder = PoseEncoderConv(args.vae_test_len, args.vae_test_dim, feature_length=args.vae_length)
        self.decoder = PoseDecoderConv(args.vae_test_len, args.vae_test_dim, feature_length=args.vae_length)
 
class VAELSTM(VAEConv):
    def __init__(self, args):
        super(VAELSTM, self).__init__(args)
        pose_dim = args.vae_test_dim
        feature_length = args.vae_length
        self.encoder = PoseEncoderLSTM_Resnet(pose_dim, feature_length=feature_length)
        self.decoder = PoseDecoderLSTM(pose_dim, feature_length=feature_length)

class VAETransformer(VAEConv):
    def __init__(self, args):
        super(VAETransformer, self).__init__(args)
        self.encoder = Encoder_TRANSFORMER(args)
        self.decoder = Decoder_TRANSFORMER(args)

# ----------- VQVAE --------------- #
class VQVAEConv(nn.Module):
    def __init__(self, args):
        super(VQVAEConv, self).__init__()
        self.encoder = VQEncoderV3(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV3(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
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

class VQVAESKConv(VQVAEConv):
    def __init__(self, args):
        super(VQVAESKConv, self).__init__(args)
        smpl_fname = args.data_path_1+'smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'
        smpl_data = np.load(smpl_fname, encoding='latin1')
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)
        self.encoder = LocalEncoder(args, edges)


class VQVAEConvStride(nn.Module):
    def __init__(self, args):
        super(VQVAEConvStride, self).__init__()
        self.encoder = VQEncoderV4(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV4(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
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

class VQVAEConvZero(nn.Module):
    def __init__(self, args):
        super(VQVAEConvZero, self).__init__()
        self.encoder = VQEncoderV5(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
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
    

class VAEConvZero(nn.Module):
    def __init__(self, args):
        super(VAEConvZero, self).__init__()
        self.encoder = VQEncoderV5(args)
        # self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        # embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(pre_latent)
        return {
            # "poses_feat":vq_latent,
            # "embedding_loss":embedding_loss,
            # "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    # def map2index(self, inputs):
    #     pre_latent = self.encoder(inputs)
    #     index = self.quantizer.map2index(pre_latent)
    #     return index
    
    # def map2latent(self, inputs):
    #     pre_latent = self.encoder(inputs)
    #     index = self.quantizer.map2index(pre_latent)
    #     z_q = self.quantizer.get_codebook_entry(index)
    #     return z_q
    
    # def decode(self, index):
    #     z_q = self.quantizer.get_codebook_entry(index)
    #     rec_pose = self.decoder(z_q)
    #     return rec_pose


class VQVAEConvZero3(nn.Module):
    def __init__(self, args):
        super(VQVAEConvZero3, self).__init__()
        self.encoder = VQEncoderV5(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
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

class VQVAEConvZero2(nn.Module):
    def __init__(self, args):
        super(VQVAEConvZero2, self).__init__()
        self.encoder = VQEncoderV5(args)
        self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV7(args)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
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

class VQVAE2(nn.Module):
    def __init__(self, args):
        super(VQVAE2, self).__init__()
        # Bottom-level encoder and decoder
        args_bottom = copy.deepcopy(args)
        args_bottom.vae_layer = 2
        self.bottom_encoder = VQEncoderV6(args_bottom)
        self.bottom_quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        args_bottom.vae_test_dim = args.vae_test_dim
        self.bottom_decoder = VQDecoderV6(args_bottom)
        
        # Top-level encoder and decoder
        args_top = copy.deepcopy(args)
        args_top.vae_layer = 3
        args_top.vae_test_dim = args.vae_length
        self.top_encoder = VQEncoderV3(args_top)  # Adjust according to the top level's design
        self.quantize_conv_t = nn.Conv1d(args.vae_length+args.vae_length, args.vae_length, 1)
        self.top_quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        # self.upsample_t_up = nn.Upsample(scale_factor=2, mode='nearest')
        layers = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(args.vae_length, args.vae_length, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(args.vae_length, args.vae_length, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(args.vae_length, args.vae_length, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.upsample_t= nn.Sequential(*layers)
        self.top_decoder = VQDecoderV3(args_top)  # Adjust to handle top level features appropriately

    def forward(self, inputs):
        # Bottom-level processing
        enc_b = self.bottom_encoder(inputs)
        enc_t = self.top_encoder(enc_b)
        #print(enc_b.shape, enc_t.shape)
        top_embedding_loss, quant_t, _, top_perplexity = self.top_quantizer(enc_t)
        #print(quant_t.shape)
        dec_t = self.top_decoder(quant_t)
        #print(dec_t.shape)
        enc_b = torch.cat([dec_t, enc_b], dim=2).permute(0,2,1)
        #print(enc_b.shape)
        quant_b = self.quantize_conv_t(enc_b).permute(0,2,1)
        #print("5",quant_b.shape)
        bottom_embedding_loss, quant_b, _, bottom_perplexity = self.bottom_quantizer(quant_b)
        #print("6",quant_b.shape)
        upsample_t = self.upsample_t(quant_t.permute(0,2,1)).permute(0,2,1)
        #print("7",upsample_t.shape)
        quant = torch.cat([upsample_t, quant_b], 2)
        rec_pose = self.bottom_decoder(quant)
        # print(quant_t.shape, quant_b.shape, rec_pose.shape)
        return {
            "poses_feat_top": quant_t,
            "pose_feat_bottom": quant_b,
            "embedding_loss":top_embedding_loss+bottom_embedding_loss,
            #"perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    def map2index(self, inputs):
        enc_b = self.bottom_encoder(inputs)
        enc_t = self.top_encoder(enc_b)
        
        _, quant_t, _, _ = self.top_quantizer(enc_t)
        top_index = self.top_quantizer.map2index(enc_t)
        dec_t = self.top_decoder(quant_t)

        enc_b = torch.cat([dec_t, enc_b], dim=2).permute(0,2,1)
        #print(enc_b.shape)
        quant_b = self.quantize_conv_t(enc_b).permute(0,2,1)
        # quant_b = self.quantize_conv_t(enc_b)
        bottom_index = self.bottom_quantizer.map2index(quant_b)
        return top_index, bottom_index
    
    def get_top_laent(self, top_index):
        z_q_top = self.top_quantizer.get_codebook_entry(top_index)
        return z_q_top
    
    def map2latent(self, inputs):
        enc_b = self.bottom_encoder(inputs)
        enc_t = self.top_encoder(enc_b)
        
        _, quant_t, _, _ = self.top_quantizer(enc_t)
        top_index = self.top_quantizer.map2index(enc_t)
        dec_t = self.top_decoder(quant_t)

        enc_b = torch.cat([dec_t, enc_b], dim=2).permute(0,2,1)
        #print(enc_b.shape)
        quant_b = self.quantize_conv_t(enc_b).permute(0,2,1)
        # quant_b = self.quantize_conv_t(enc_b)
        bottom_index = self.bottom_quantizer.map2index(quant_b)
        z_q_top = self.top_quantizer.get_codebook_entry(top_index)
        z_q_bottom = self.bottom_quantizer.get_codebook_entry(bottom_index)
        return z_q_top, z_q_bottom
    
    def map2latent_top(self, inputs):
        enc_b = self.bottom_encoder(inputs)
        enc_t = self.top_encoder(enc_b)
        top_index = self.top_quantizer.map2index(enc_t)
        z_q_top = self.top_quantizer.get_codebook_entry(top_index)
        return z_q_top
    
    def decode(self, top_index, bottom_index):
        quant_t = self.top_quantizer.get_codebook_entry(top_index)
        quant_b = self.bottom_quantizer.get_codebook_entry(bottom_index)
        upsample_t = self.upsample_t(quant_t.permute(0,2,1)).permute(0,2,1)
        #print("7",upsample_t.shape)
        quant = torch.cat([upsample_t, quant_b], 2)
        rec_pose = self.bottom_decoder(quant)      
        return rec_pose