import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )
    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim, feature_length=32):
        super().__init__()
        self.base = feature_length
        self.net = nn.Sequential(
            ConvNormRelu(dim, self.base, batchnorm=True), #32
            ConvNormRelu(self.base, self.base*2, batchnorm=True), #30
            ConvNormRelu(self.base*2, self.base*2, True, batchnorm=True), #14
            nn.Conv1d(self.base*2, self.base, 3)
        )
        self.out_net = nn.Sequential(
            nn.Linear(12*self.base, self.base*4),  # for 34 frames
            nn.BatchNorm1d(self.base*4),
            nn.LeakyReLU(True),
            nn.Linear(self.base*4, self.base*2),
            nn.BatchNorm1d(self.base*2),
            nn.LeakyReLU(True),
            nn.Linear(self.base*2, self.base),
        )

        self.fc_mu = nn.Linear(self.base, self.base)
        self.fc_logvar = nn.Linear(self.base, self.base)

    def forward(self, poses, variational_encoding=None):
        # encode
        poses = poses.transpose(1, 2) 
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False, feature_length=32):
        super().__init__()
        self.use_pre_poses = use_pre_poses
        self.feat_size = feature_length
        
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            self.feat_size += 32

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(True),
                nn.Linear(128, 256),
            )
        elif length == 34:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size*2),
                nn.BatchNorm1d(self.feat_size*2),
                nn.LeakyReLU(True),
                nn.Linear(self.feat_size*2, self.feat_size//8*34),
            )
        else:
            assert False
        self.decoder_size = self.feat_size//8
        self.net = nn.Sequential(
            nn.ConvTranspose1d(self.decoder_size, self.feat_size, 3),
            nn.BatchNorm1d(self.feat_size),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose1d(self.feat_size, self.feat_size, 3),
            nn.BatchNorm1d(self.feat_size),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(self.feat_size, self.feat_size*2, 3),
            nn.Conv1d(self.feat_size*2, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)
        #print(feat.shape)
        out = self.pre_net(feat)
        #print(out.shape)
        out = out.view(feat.shape[0], self.decoder_size, -1)
        #print(out.shape)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out

    
class EmbeddingNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_frames = args.pose_length
        pose_dim = args.pose_dims
        feature_length = args.vae_length
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim, feature_length=feature_length)
        self.decoder = PoseDecoderConv(n_frames, pose_dim, feature_length=feature_length)

    def forward(self, pre_poses, poses, variational_encoding=False):
        poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
        latent_feat = poses_feat
        out_poses = self.decoder(latent_feat, pre_poses)
        return poses_feat, pose_mu, pose_logvar, out_poses

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

            
class HalfEmbeddingNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_frames = args.pose_length
        pose_dim = args.pose_dims
        feature_length = args.vae_length
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim, feature_length=feature_length)
        self.decoder = PoseDecoderConv(n_frames, pose_dim, feature_length=feature_length)

    def forward(self, poses):
        poses_feat, _, _ = self.pose_encoder(poses)
        return poses_feat