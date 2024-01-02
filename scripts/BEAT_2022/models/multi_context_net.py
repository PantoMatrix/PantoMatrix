import math
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
from torch.nn.utils import weight_norm
from .utils.build_vocab import Vocab


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None: 
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layer
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], args.word_f)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


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
                BasicBlock(1, 32, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(32, 32, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(32, 32, 15, 1, first_dilation=7, ),
                BasicBlock(32, 64, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(64, 64, 15, 1, first_dilation=7),
                BasicBlock(64, 128, 15, 6,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  


class PoseGenerator(nn.Module):
    """
    reimplement: 
         Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity,
         Yoon et al. SIGGRAPH ASIA 2020,
         https://arxiv.org/abs/2009.02119,
         https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context,
    with resnet based audio encoder
    """
    def __init__(self, args):
        super().__init__()
        self.pre_length = args.pre_frames 
        self.gen_length = args.pose_length - args.pre_frames 
        self.pose_dims = args.pose_dims
        self.facial_f = args.facial_f
        self.speaker_f = args.speaker_f
        self.audio_f = args.audio_f
        self.word_f = args.word_f
        self.emotion_f = args.emotion_f
        self.facial_dims = args.facial_dims
        self.speaker_dims = args.speaker_dims
        self.emotion_dims = args.emotion_dims
        self.word_index_num = args.word_index_num
        self.word_dims = args.word_dims
        self.in_size = self.audio_f + self.pose_dims + self.word_f + 1
        self.audio_encoder = WavEncoder(self.audio_f)
        
        self.hidden_size = args.hidden_size
        self.n_layer = args.n_layer

        self.text_encoder = None   
        if self.word_f is not 0:
            self.text_encoder = TextEncoderTCN(args, self.word_index_num, self.word_dims, pre_trained_embedding=None,
                                           dropout=args.dropout_prob)
        self.speaker_embedding = None
        if self.speaker_f is not 0:
            self.in_size += self.speaker_f
            self.speaker_embedding = nn.Sequential(
                nn.Embedding(self.speaker_dims, self.speaker_f),
                nn.Linear(self.speaker_f, self.speaker_f), 
                nn.LeakyReLU(True)
            )
            self.speaker_mu = nn.Linear(self.speaker_f, self.speaker_f)
            self.speaker_logvar = nn.Linear(self.speaker_f, self.speaker_f)
        
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layer, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size//2, self.pose_dims)
        )
        
        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std        

    def forward(self, pre_seq, in_audio=None, in_word=None, in_id=None):
        hidden = speaker_feat_seq = z_mu = z_logvar = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        audio_feat_seq = self.audio_encoder(in_audio)  
        text_feat_seq, _ = self.text_encoder(in_word)
        if audio_feat_seq.shape[1] != text_feat_seq.shape[1]:
            diff_length = text_feat_seq.shape[1] - audio_feat_seq.shape[1]
            audio_feat_seq = torch.cat((audio_feat_seq, audio_feat_seq[:,-diff_length:, :].reshape(1,diff_length,-1)),1)

        speaker_feat_seq = None
        if self.speaker_embedding: 
            speaker_feat_seq = self.speaker_embedding(in_id)
            z_mu = self.speaker_mu(speaker_feat_seq)
            z_logvar = self.speaker_logvar(speaker_feat_seq)
            speaker_feat_seq = self.reparameterize(z_mu, z_logvar)
    
        if  audio_feat_seq.shape[1] != pre_seq.shape[1]:
            diff_length = pre_seq.shape[1] - audio_feat_seq.shape[1]
            audio_feat_seq = torch.cat((audio_feat_seq, audio_feat_seq[:,-diff_length:, :].reshape(1,diff_length,-1)),1)
        in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        in_data = torch.cat((in_data, text_feat_seq), dim=2)
        
        if speaker_feat_seq is not None:
            repeated_s = speaker_feat_seq
            if len(repeated_s.shape) == 2:
                repeated_s = repeated_s.reshape(1, repeated_s.shape[1], repeated_s.shape[0])
            repeated_s = repeated_s.repeat(1, in_data.shape[1], 1)
            in_data = torch.cat((in_data, repeated_s), dim=2)
        
        output, hidden = self.gru(in_data, hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)
        return decoder_outputs, speaker_feat_seq, z_mu, z_logvar
    

class ConvDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.pose_dims
        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(self.input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.Conv1d(8, 8, 3),
        )

        self.gru = nn.GRU(8, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(34-6, 1) 
       
        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()
        poses = poses.transpose(1, 2)
        feat = self.pre_conv(poses)
        feat = feat.transpose(1, 2)
        output, decoder_hidden = self.gru(feat, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)
        return output