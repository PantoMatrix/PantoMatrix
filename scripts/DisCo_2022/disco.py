import torch
import torch.nn as nn
import os
import math
import pickle
import numpy as np
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ based on timm: https://github.com/rwightman/pytorch-image-models """
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU,   norm_layer=nn.BatchNorm1d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,  stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes), 
            )
        else: self.downsample=None
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

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
    def __init__(self, out_dim):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(1, out_dim//4, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 6,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1) 
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2) 

    
class LSTMMLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, out_dim)
        )
        self.hidden_size = hidden_size
    def forward(self, inputs):
        out, hidden = self.lstm(inputs)
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        out = self.mlp(out)
        hidden = torch.mean(hidden[0],dim=0)#avgpooling
        #print(hidden.shape)
        return out, hidden

    
class LP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
    def forward(self, inputs):
        out = self.mlp(inputs) #bs*n*128
        return out, torch.max(out, dim=1)[0] 
    
class Empty(nn.Module):
    def __init__(self):
        super().__init__()
        #self.mlp = nn.Linear(in_dim, out_dim)
    def forward(self, inputs):
        #out = self.mlp(inputs) #bs*n*128
        return inputs
    

class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.a_linear = self.a_linear_m = self.t_linear = self.t_linear_m = self.m_linear = self.m_linear_m =  None
        
        # ----------------- audio ------------------------------- #
        self.audio_pre_encoder = WavEncoder(args.audio_f)
        self.audio_encoder_c1 = LSTMMLP(args.audio_f, args.audio_f, args.audio_f, args.n_layer, args.dropout_prob)
        self.audio_encoder_c2 = LSTMMLP(args.audio_f, args.audio_f, args.audio_f, args.n_layer, args.dropout_prob)
        self.audio_encoder_r = LSTMMLP(args.audio_f, args.audio_f, args.audio_f, args.n_layer, args.dropout_prob)
        
        # ------------------------ motion ---------------------------------- #
        self.motion_pre_encoder = Empty()
        
        # ----------------------- motion decoder --------------------- #
        if args.m_decoder == "lstm":
            if "cat" in args.decode_fusion:
                decode_f = motion_in_f+args.audio_f+args.word_f
            self.motion_decoder = LSTMMLP(decode_f, args.hidden_size, args.pose_dims, args.n_layer, args.dropout_prob)
            
        # ----------------------- contrastive ------------------- # 
        if "unsupervised" in args.decode_fusion:
            self.selector = LSTMMLP(args.audio_f, args.hidden_size, 2, args.n_layer, args.dropout_prob)
            self.softmax = nn.Softmax(dim=2)
         
        self.decode_fusion = args.decode_fusion
        self.m_decoder = args.m_decoder
    
    def select_content(self, audio_f):
        sum_f = audio_f + text_f # bs * n * 128
        reduce_f, _ = self.selector(sum_f)
        weights_f = self.softmax(reduce_f) # bs * n * 2
        return weights_f
    
    def get_weights_from_gt(self, gt_label):
        weights_c1 = torch.where(gt_label<=0.2, 1.0, 0.0).unsqueeze(2)#bs*n
        weights_c2 = torch.where(gt_label<=0.2, 0.0, 1.0).unsqueeze(2)#bs*n
        weights_f = torch.cat((weights_c1, weights_c2), 2)
        return weights_f 
    
    def forward(self, pre_seq, in_audio=None, is_test="test", in_gt=None, weights_f=None):
        audio_feat_seq = self.audio_pre_encoder(in_audio) if self.a_linear is None else self.a_linear(self.audio_pre_encoder(in_audio))
       
        audio_feat_seq_with_pad_r, cls_r = self.audio_encoder_r(audio_feat_seq) # bs * 256
        audio_feat_seq_with_pad_c1, cls_c1 = self.audio_encoder_c1(audio_feat_seq)
        audio_feat_seq_with_pad_c2, cls_c2 = self.audio_encoder_c2(audio_feat_seq)
        
        if audio_feat_seq_with_pad.shape[1] != pre_seq.shape[1]:
            diff_length = pre_seq.shape[1] - audio_feat_seq_with_pad.shape[1]
            audio_feat_seq_with_pad = torch.cat((audio_feat_seq_with_pad, audio_feat_seq_with_pad[:,-diff_length:, :].reshape(audio_feat_seq_with_pad.shape[0],diff_length,-1)),1)
    
        if weights_f is None:
            if "unsupervised" in self.decode_fusion:
                weights_f = self.select_content(audio_feat_seq)
                if is_test == "test":
                    np.save(f"{self.weight_save}{self.test_counter}.npy", weights_f.detach().cpu().numpy())
                    self.test_counter += 1
            else:
                weights_f = self.get_weights_from_gt(in_gt)
        else:
            weights_f_real = weights_f[:, :audio_feat_seq_with_pad.shape[1], :].to(audio_feat_seq_with_pad.device)
            weights_f = weights_f_real
           
            
        if "cat" in self.decode_fusion:
            fusion_feat_seq = audio_feat_seq_with_pad_c1 * weights_f[:, :, 0:1] + audio_feat_seq_with_pad_c2 * weights_f[:, :, 1:2]
      
        pre_motion_feat_seq = self.motion_pre_encoder(pre_seq) if self.m_linear is None else self.m_linear(self.motion_pre_encoder(pre_seq))
        fusion_feat_seq = torch.cat((pre_motion_feat_seq, fusion_feat_seq, audio_feat_seq_with_pad_r), dim=2)
        output, _ = self.motion_decoder(fusion_feat_seq)
            
        if is_test == "train":
            return {"audio_feat_seq_r": cls_r,
                    "audio_feat_seq_c1": cls_c1, 
                    "audio_feat_seq_c2": cls_c2, 
                    "rec_pose":output, 
                    "weights_f":weights_f,}
        elif is_test == "val":
            return {"rec_pose":output, 
                    "weights_f":weights_f,}
        else:
            return {"rec_pose":output}