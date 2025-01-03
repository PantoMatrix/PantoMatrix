import torch.nn as nn
import torch
import numpy as np
from .skeleton_DME import SkeletonConv, SkeletonPool, find_neighbor, build_edge_topology
from .skeleton import SkeletonResidual
from .decoders import VQDecoderV3


class LocalEncoder(nn.Module):
    def __init__(self, args, topology):
        super(LocalEncoder, self).__init__()
        args.channel_base = 6
        args.activation = "tanh"
        args.use_residual_blocks = True
        args.z_dim = 1024
        args.temporal_scale = 8
        args.kernel_size = 4
        args.num_layers = args.vae_layer
        args.skeleton_dist = 2
        args.extra_conv = 0
        # check how to reflect in 1d
        args.padding_mode = "constant"
        args.skeleton_pool = "mean"
        args.upsampling = "linear"

        self.topologies = [topology]
        self.channel_base = [args.channel_base]

        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        # self.convs = []

        kernel_size = args.kernel_size
        kernel_even = False if kernel_size % 2 else True
        padding = (kernel_size - 1) // 2
        bias = True
        self.grow = args.vae_grow
        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * self.grow[i])

        for i in range(args.num_layers):
            seq = []
            neighbour_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i + 1] * self.edge_num[i]
            if i == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            last_pool = True if i == args.num_layers - 1 else False

            # (T, J, D) => (T, J', D)
            pool = SkeletonPool(
                edges=self.topologies[i],
                pooling_mode=args.skeleton_pool,
                channels_per_edge=out_channels // len(neighbour_list),
                last_pool=last_pool,
            )

            if args.use_residual_blocks:
                # (T, J, D) => (T/2, J', 2D)
                seq.append(
                    SkeletonResidual(
                        self.topologies[i],
                        neighbour_list,
                        joint_num=self.edge_num[i],
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        padding_mode=args.padding_mode,
                        bias=bias,
                        extra_conv=args.extra_conv,
                        pooling_mode=args.skeleton_pool,
                        activation=args.activation,
                        last_pool=last_pool,
                    )
                )
            else:
                for _ in range(args.extra_conv):
                    # (T, J, D) => (T, J, D)
                    seq.append(
                        SkeletonConv(
                            neighbour_list,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            joint_num=self.edge_num[i],
                            kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                            stride=1,
                            padding=padding,
                            padding_mode=args.padding_mode,
                            bias=bias,
                        )
                    )
                    seq.append(nn.PReLU() if args.activation == "relu" else nn.Tanh())
                # (T, J, D) => (T/2, J, 2D)
                seq.append(
                    SkeletonConv(
                        neighbour_list,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        joint_num=self.edge_num[i],
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        padding_mode=args.padding_mode,
                        bias=bias,
                        add_offset=False,
                        in_offset_channel=3 * self.channel_base[i] // self.channel_base[0],
                    )
                )
                # self.convs.append(seq[-1])

                seq.append(pool)
                seq.append(nn.PReLU() if args.activation == "relu" else nn.Tanh())
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

        # in_features = self.channel_base[-1] * len(self.pooling_list[-1])
        # in_features *= int(args.temporal_scale / 2)
        # self.reduce = nn.Linear(in_features, args.z_dim)
        # self.mu = nn.Linear(in_features, args.z_dim)
        # self.logvar = nn.Linear(in_features, args.z_dim)

    def forward(self, input):
        # bs, n, c = input.shape[0], input.shape[1], input.shape[2]
        output = input.permute(0, 2, 1)  # input.reshape(bs, n, -1, 6)
        for layer in self.layers:
            output = layer(output)
        # output = output.view(output.shape[0], -1)
        output = output.permute(0, 2, 1)
        return output


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAEConv(nn.Module):
    def __init__(self, args):
        super(VAEConv, self).__init__()
        # self.encoder = VQEncoderV3(args)
        # self.decoder = VQDecoderV3(args)
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
            "poses_feat": pre_latent,
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
    def __init__(self, args, model_save_path="./emage/"):
        # args = args()
        super(VAESKConv, self).__init__(args)
        smpl_fname = model_save_path + "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
        smpl_data = np.load(smpl_fname, encoding="latin1")
        parents = smpl_data["kintree_table"][0].astype(np.int32)
        edges = build_edge_topology(parents)
        self.encoder = LocalEncoder(args, edges)
        self.decoder = VQDecoderV3(args)
