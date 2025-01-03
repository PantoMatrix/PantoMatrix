import torch
import torch.nn as nn

from .skeleton_DME import SkeletonConv, SkeletonPool, SkeletonUnpool


def calc_node_depth(topology):
    def dfs(node, topology):
        if topology[node] < 0:
            return 0
        return 1 + dfs(topology[node], topology)

    depth = []
    for i in range(len(topology)):
        depth.append(dfs(i, topology))

    return depth


def residual_ratio(k):
    return 1 / (k + 1)


class Affine(nn.Module):
    def __init__(self, num_parameters, scale=True, bias=True, scale_init=1.0):
        super(Affine, self).__init__()
        if scale:
            self.scale = nn.Parameter(torch.ones(num_parameters) * scale_init)
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(num_parameters))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        output = input
        if self.scale is not None:
            scale = self.scale.unsqueeze(0)
            while scale.dim() < input.dim():
                scale = scale.unsqueeze(2)
        output = output.mul(scale)

        if self.bias is not None:
            bias = self.bias.unsqueeze(0)
            while bias.dim() < input.dim():
                bias = bias.unsqueeze(2)
        output += bias

        return output


class BatchStatistics(nn.Module):
    def __init__(self, affine=-1):
        super(BatchStatistics, self).__init__()
        self.affine = nn.Sequential() if affine == -1 else Affine(affine)
        self.loss = 0

    def clear_loss(self):
        self.loss = 0

    def compute_loss(self, input):
        input_flat = input.view(input.size(1), input.numel() // input.size(1))
        mu = input_flat.mean(1)
        logvar = (input_flat.pow(2).mean(1) - mu.pow(2)).sqrt().log()

        self.loss = mu.pow(2).mean() + logvar.pow(2).mean()

    def forward(self, input):
        self.compute_loss(input)
        return self.affine(input)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, residual_ratio, activation, batch_statistics=False, last_layer=False
    ):
        super(ResidualBlock, self).__init__()

        self.residual_ratio = residual_ratio
        self.shortcut_ratio = 1 - residual_ratio

        residual = []
        residual.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
        if batch_statistics:
            residual.append(BatchStatistics(out_channels))
        if not last_layer:
            residual.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        self.residual = nn.Sequential(*residual)

        self.shortcut = nn.Sequential(
            nn.AvgPool1d(kernel_size=2) if stride == 2 else nn.Sequential(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            BatchStatistics(out_channels) if (in_channels != out_channels and batch_statistics is True) else nn.Sequential(),
        )

    def forward(self, input):
        return self.residual(input).mul(self.residual_ratio) + self.shortcut(input).mul(self.shortcut_ratio)


class ResidualBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, residual_ratio, activation):
        super(ResidualBlockTranspose, self).__init__()

        self.residual_ratio = residual_ratio
        self.shortcut_ratio = 1 - residual_ratio

        self.residual = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding), nn.PReLU() if activation == "relu" else nn.Tanh()
        )

        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False) if stride == 2 else nn.Sequential(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input):
        return self.residual(input).mul(self.residual_ratio) + self.shortcut(input).mul(self.shortcut_ratio)


class SkeletonResidual(nn.Module):
    def __init__(
        self,
        topology,
        neighbour_list,
        joint_num,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        bias,
        extra_conv,
        pooling_mode,
        activation,
        last_pool,
    ):
        super(SkeletonResidual, self).__init__()

        kernel_even = False if kernel_size % 2 else True

        seq = []
        for _ in range(extra_conv):
            # (T, J, D) => (T, J, D)
            seq.append(
                SkeletonConv(
                    neighbour_list,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    joint_num=joint_num,
                    kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                    stride=1,
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=bias,
                )
            )
            seq.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        # (T, J, D) => (T/2, J, 2D)
        seq.append(
            SkeletonConv(
                neighbour_list,
                in_channels=in_channels,
                out_channels=out_channels,
                joint_num=joint_num,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias,
                add_offset=False,
            )
        )
        seq.append(nn.GroupNorm(10, out_channels))  # FIXME: REMEMBER TO CHANGE BACK !!!
        self.residual = nn.Sequential(*seq)

        # (T, J, D) => (T/2, J, 2D)
        self.shortcut = SkeletonConv(
            neighbour_list,
            in_channels=in_channels,
            out_channels=out_channels,
            joint_num=joint_num,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=True,
            add_offset=False,
        )

        seq = []
        # (T/2, J, 2D) => (T/2, J', 2D)
        pool = SkeletonPool(
            edges=topology, pooling_mode=pooling_mode, channels_per_edge=out_channels // len(neighbour_list), last_pool=last_pool
        )
        if len(pool.pooling_list) != pool.edge_num:
            seq.append(pool)
        seq.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        self.common = nn.Sequential(*seq)

    def forward(self, input):
        output = self.residual(input) + self.shortcut(input)

        return self.common(output)


class SkeletonResidualTranspose(nn.Module):
    def __init__(
        self,
        neighbour_list,
        joint_num,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        padding_mode,
        bias,
        extra_conv,
        pooling_list,
        upsampling,
        activation,
        last_layer,
    ):
        super(SkeletonResidualTranspose, self).__init__()

        kernel_even = False if kernel_size % 2 else True

        seq = []
        # (T, J, D) => (2T, J, D)
        if upsampling is not None:
            seq.append(nn.Upsample(scale_factor=2, mode=upsampling, align_corners=False))
        # (2T, J, D) => (2T, J', D)
        unpool = SkeletonUnpool(pooling_list, in_channels // len(neighbour_list))
        if unpool.input_edge_num != unpool.output_edge_num:
            seq.append(unpool)
        self.common = nn.Sequential(*seq)

        seq = []
        for _ in range(extra_conv):
            # (2T, J', D) => (2T, J', D)
            seq.append(
                SkeletonConv(
                    neighbour_list,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    joint_num=joint_num,
                    kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                    stride=1,
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=bias,
                )
            )
            seq.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        # (2T, J', D) => (2T, J', D/2)
        seq.append(
            SkeletonConv(
                neighbour_list,
                in_channels=in_channels,
                out_channels=out_channels,
                joint_num=joint_num,
                kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias,
                add_offset=False,
            )
        )
        self.residual = nn.Sequential(*seq)

        # (2T, J', D) => (2T, J', D/2)
        self.shortcut = SkeletonConv(
            neighbour_list,
            in_channels=in_channels,
            out_channels=out_channels,
            joint_num=joint_num,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            add_offset=False,
        )

        if activation == "relu":
            self.activation = nn.PReLU() if not last_layer else None
        else:
            self.activation = nn.Tanh() if not last_layer else None

    def forward(self, input):
        output = self.common(input)
        output = self.residual(output) + self.shortcut(output)

        if self.activation is not None:
            return self.activation(output)
        else:
            return output
