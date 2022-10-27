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
#         self.block_list = [
#             [dim, self.base, False, 0, True],
#             [self.base, self.base*2, False, 0, True],
#             [self.base*2, self.base*4, True, 0, True], #14
#             [self.base*4, self.base*4, False, 0, True], #12
#             [self.base*4, self.base*8, True, 0, True], # 5
#             [self.base*8, self.base*8, True, 0, True], 
#         ]
        self.net = nn.Sequential(
            ConvNormRelu(dim, self.base, batchnorm=True), #32
            ConvNormRelu(self.base, self.base*2, batchnorm=True), #30
            ConvNormRelu(self.base*2, self.base*2, True, batchnorm=True), #14
#             ConvNormRelu(self.base*2, self.base*4, True, batchnorm=True), #6 
#             ConvNormRelu(self.base*4, self.base*8, True, batchnorm=True), #2 
            
            nn.Conv1d(self.base*2, self.base, 3)
        )

        self.out_net = nn.Sequential(
            # nn.Linear(864, 256),  # for 64 frames
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
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class PoseDecoderFC(nn.Module):
    def __init__(self, gen_length, pose_dim, use_pre_poses=False):
        super().__init__()
        self.gen_length = gen_length
        self.pose_dim = pose_dim
        self.use_pre_poses = use_pre_poses

        in_size = 32
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(pose_dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            in_size += 32

        self.net = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, gen_length * pose_dim),
        )

    def forward(self, latent_code, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, latent_code), dim=1)
        else:
            feat = latent_code
        output = self.net(feat)
        output = output.view(-1, self.gen_length, self.pose_dim)

        return output


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

    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU,   norm_layer=nn.BatchNorm1d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
#         first_dilation = first_dilation or dilation
#         use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv1d(
            inplanes, first_planes, kernel_size=3, stride=stride, padding=first_dilation,
            dilation=first_dilation, bias=True)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        #self.aa = aa_layer(channels=first_planes, stride=stride) if use_aa else None

        self.conv2 = nn.Conv1d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=True)
        self.bn2 = norm_layer(outplanes)

        #self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, outplanes,  stride=stride, kernel_size=3, padding=dilation, dilation=dilation, bias=True),
                norm_layer(outplanes), 
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
        #print("x after 1", x.shape)
#         if self.drop_block is not None:
#             x = self.drop_block(x)
        x = self.act1(x)
#         if self.aa is not None:
#             x = self.aa(x)

        x = self.conv2(x)
        #print("x after 2", x.shape)
        x = self.bn2(x)
#         if self.drop_block is not None:
#             x = self.drop_block(x)

#         if self.se is not None:
#             x = self.se(x)

#         if self.drop_path is not None:
#             x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        #print("x after 3", x.shape)
        return x

    
class PoseEncoderConvResNet(nn.Module):
    def __init__(self, length, dim, feature_length=32):
        super().__init__()
        self.base = feature_length
#         self.block_list = [
#             [dim, self.base, False, 0, True],
#             [self.base, self.base*2, False, 0, True],
#             [self.base*2, self.base*4, True, 0, True], #14
#             [self.base*4, self.base*4, False, 0, True], #12
#             [self.base*4, self.base*8, True, 0, True], # 5
#             [self.base*8, self.base*8, True, 0, True], 
#         ]
#         self.net = nn.Sequential(
#             BasicBlock(dim, self.base, reduce_first = 1, downsample = False, first_dilation=1), #32
#             BasicBlock(self.base, self.base*2, downsample = False, first_dilation=1,), #30
#             BasicBlock(self.base*2, self.base*2, first_dilation=1, downsample = True), #14            
#             BasicBlock(self.base*2, self.base, first_dilation=1, downsample = False),
#         )
        self.conv1=BasicBlock(dim, self.base, reduce_first = 1, downsample = False, first_dilation=1) #34
        self.conv2=BasicBlock(self.base, self.base*2, downsample = False, first_dilation=1,) #34
        self.conv3=BasicBlock(self.base*2, self.base*2, first_dilation=1, downsample = True, stride=2)#17            
        self.conv4=BasicBlock(self.base*2, self.base, first_dilation=1, downsample = False)
        

        self.out_net = nn.Sequential(
            # nn.Linear(864, 256),  # for 64 frames
            nn.Linear(17*self.base, self.base*4),  # for 34 frames
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
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
#         out = self.net(poses)
        out1 = self.conv1(poses)
        #print(out1.shape)
        out2 = self.conv2(out1)
        #print(out2.shape)
        out3 = self.conv3(out2)
        #print(out3.shape)
        out = self.conv4(out3)
        #print(out.shape)
        out = out.flatten(1)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar    
    
    
class PoseEncoderGRU_Resnet(nn.Module):
    """
    input: n * 123 (15fps)
    output n * 32 (feature_dim)
    field : should be at least 34(32)
    """
    def __init__(self, pose_dim, feature_length):
        super().__init__()
        #self.gen_length = gen_length
        self.pose_dim = pose_dim
        self.base = feature_length
        self.hidden_size = 64# 300
        
        self.conv1 = BasicBlock(self.pose_dim, self.base, reduce_first = 1, downsample = True, first_dilation=1) # 123 * n -> 64 * (n-2) padding = same
#         self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
#                  reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU,   norm_layer=nn.BatchNorm1d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None
        self.conv2 = BasicBlock(self.base, self.base*2, downsample = True, first_dilation=1,)
        self.conv3 = BasicBlock(self.base*2, self.base*2, first_dilation=1,)
        self.conv4 = BasicBlock(self.base*2, self.base*2, first_dilation=1,)
        
        self.gru = nn.GRU(self.base*2, hidden_size=self.hidden_size, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, self.base*2)
        )

    def forward(self, poses):
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        #print(poses.shape)
        out = self.conv1(poses)
        out = self.conv2(out)
        
        out = self.conv3(out)
        out = self.conv4(out)
#         out = self.conv5(out)
#         out = self.conv6(out)
#         out = self.conv7(out)
#         out = self.conv8(out) # 64 * n
        #print("after conv:", out.shape)
        out = out.transpose(1, 2) # 128*64*34 to 128*34*64
#         pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
#         feat = torch.cat((pre_pose_feat, latent_code), dim=1)
#         feat = feat.unsqueeze(1).repeat(1, self.gen_length, 1)

        output, decoder_hidden = self.gru(out)
        #print("out", output.shape, "hidden", decoder_hidden.shape)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs # 300*n
        # 128 * 34 * 300
        output = self.out(output.reshape(-1, output.shape[2])) 
        output = output.view(poses.shape[0], poses.shape[2], -1) #
        #print("after encoder:",output.shape)
        return output     
    
    
class PoseEncoderGRU(nn.Module):
    """
    input: n * 123 (15fps)
    output n * 32 (feature_dim)
    field : should be at least 34(32)
    """
    def __init__(self, pose_dim, feature_length):
        super().__init__()
        #self.gen_length = gen_length
        self.pose_dim = pose_dim
        self.base = feature_length
        self.hidden_size = 300# 300
        
        
        self.conv1 = ConvNormRelu(self.pose_dim, self.base, padding=1, batchnorm=True) # 123 * n -> 64 * (n-2) padding = same
        self.conv2 = ConvNormRelu(self.base, self.base*2, padding=1, batchnorm=True)
        self.conv3 = ConvNormRelu(self.base*2, self.base*2, padding=1, batchnorm=True)
        self.conv4 = ConvNormRelu(self.base*2, self.base*2, padding=1, batchnorm=True)
        self.conv5 = ConvNormRelu(self.base*2, self.base*2, padding=1, batchnorm=True)
        self.conv6 = ConvNormRelu(self.base*2, self.base*2, padding=1, batchnorm=True)
        self.conv7 = ConvNormRelu(self.base*2, self.base*2, padding=1, batchnorm=True)
        self.conv8 = ConvNormRelu(self.base*2, self.base*2, padding=1, batchnorm=True) #2*8*2+1 --> 33
        
#         self.pre_pose_net = nn.Sequential(
#             nn.Linear(pose_dim * 4, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#         )
        self.gru = nn.GRU(self.base*2, hidden_size=self.hidden_size, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, self.base*2)
        )

    def forward(self, poses):
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        #print(poses.shape)
        out = self.conv1(poses)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out) # 64 * n
        #print("after conv:", out.shape)
        out = out.transpose(1, 2) # 128*64*34 to 128*34*64
#         pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
#         feat = torch.cat((pre_pose_feat, latent_code), dim=1)
#         feat = feat.unsqueeze(1).repeat(1, self.gen_length, 1)

        output, decoder_hidden = self.gru(out)
        #print("out", output.shape, "hidden", decoder_hidden.shape)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs # 300*n
        # 128 * 34 * 300
        output = self.out(output.reshape(-1, output.shape[2])) 
        output = output.view(poses.shape[0], poses.shape[2], -1) #
        #print("after encoder:",output.shape)
        return output    
    
class PoseDecoderGRU(nn.Module):
    """
    input bs*n*64
    """
    def __init__(self,pose_dim, feature_length):
        super().__init__()
        self.pose_dim = pose_dim
        self.base = feature_length
        self.hidden_size = 64

#         self.pre_pose_net = nn.Sequential(
#             nn.Linear(pose_dim * 4, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#         )
        self.gru_d = nn.GRU(self.base*2, hidden_size=self.hidden_size, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out_d = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, self.pose_dim)
        )

    def forward(self, latent_code):
        #pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
        #feat = torch.cat((pre_pose_feat, latent_code), dim=1)
        #feat = feat.unsqueeze(1).repeat(1, self.gen_length, 1)

        output, decoder_hidden = self.gru_d(latent_code)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        #print("outd:", output.shape)
        output = self.out_d(output.reshape(-1, output.shape[2]))
        output = output.view(latent_code.shape[0], latent_code.shape[1], -1)
        #print("resotuput:", output.shape)
        return output
    
    
class MotionDisNet(nn.Module):
    """ GRU motion vae"""
    def __init__(self, args):
        super().__init__()
        #n_frames = args.pose_length
        pose_dim = args.pose_dims
        feature_length = args.vae_length
        self.pose_encoder = PoseEncoderGRU_2(pose_dim, feature_length=feature_length)
        self.pose_decoder = PoseDecoderGRU(pose_dim, feature_length=feature_length)

    def forward(self, pre_poses, poses, variational_encoding=False):
        #poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses)
        #pose_mu, 
        poses_feat = self.pose_encoder(poses)
        #print(poses_feat.shape)
        latent_feat, pose_mu, pose_logvar = poses_feat, poses_feat,poses_feat
        out_poses = self.pose_decoder(latent_feat)
        return poses_feat, pose_mu, pose_logvar, out_poses

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False        
    

class EmbeddingNetResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_frames = args.pose_length
        pose_dim = args.pose_dims
        feature_length = args.vae_length
        self.pose_encoder = PoseEncoderConvResNet(n_frames, pose_dim, feature_length=feature_length)
        self.decoder = PoseDecoderConv(n_frames, pose_dim, feature_length=feature_length)

    def forward(self, pre_poses, poses, variational_encoding=False):
        poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
        #print(poses_feat.shape)
        latent_feat = poses_feat
        out_poses = self.decoder(latent_feat, pre_poses)
        return poses_feat, pose_mu, pose_logvar, out_poses

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False    
    
    
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
        #print(poses_feat.shape)
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

            

if __name__ == '__main__':
    # for model debugging
    n_frames = 34
    pose_dim = 141
    encoder = PoseEncoderConv(n_frames, pose_dim)
    decoder = PoseDecoderConv(n_frames, pose_dim)

    poses = torch.randn((4, n_frames, pose_dim))
    feat, _, _ = encoder(poses, True)
    recopose_length = decoder(feat)

    print('input', poses.shape)
    print('feat', feat.shape)
    print('output', recopose_length.shape)



    
    
    
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
        #print(poses_feat.shape)
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

            