"""Based on Daniel Holden code from:
   A Deep Learning Framework for Character Motion Synthesis and Editing
   (http://www.ipab.inf.ed.ac.uk/cgvu/motionsynthesis.pdf)
"""

import os

import numpy as np
import torch
import torch.nn as nn
from .rotations import euler_angles_to_matrix, quaternion_to_matrix, rotation_6d_to_matrix


class ForwardKinematicsLayer(nn.Module):
    """ Forward Kinematics Layer Class """

    def __init__(self, args=None, parents=None, positions=None, device=None):
        super().__init__()
        self.b_idxs = None
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if parents is None and positions is None:
            # Load SMPL skeleton (their joint order is different from the one we use for bvh export)
            smpl_fname = os.path.join(args.smpl.smpl_body_model, args.data.gender, 'model.npz')
            smpl_data = np.load(smpl_fname, encoding='latin1')
            self.parents = torch.from_numpy(smpl_data['kintree_table'][0].astype(np.int32)).to(self.device)
            self.parents = self.parents.long()
            self.positions = torch.from_numpy(smpl_data['J'].astype(np.float32)).to(self.device)
            self.positions[1:] -= self.positions[self.parents[1:]]
        else:
            self.parents = torch.from_numpy(parents).to(self.device)
            self.parents = self.parents.long()
            self.positions = torch.from_numpy(positions).to(self.device)
            self.positions = self.positions.float()
        self.positions[0] = 0

    def rotate(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

    def identity_rotation(self, rotations):
        diagonal = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0])).to(self.device)
        diagonal = torch.reshape(
            diagonal, torch.Size([1] * len(rotations.shape[:2]) + [4, 4]))
        ts = diagonal.repeat(rotations.shape[:2] + torch.Size([1, 1]))
        return ts

    def make_fast_rotation_matrices(self, positions, rotations):
        if len(rotations.shape) == 4 and rotations.shape[-2:] == torch.Size([3, 3]):
            rot_matrices = rotations
        elif rotations.shape[-1] == 3:
            rot_matrices = euler_angles_to_matrix(rotations, convention='XYZ')
        elif rotations.shape[-1] == 4:
            rot_matrices = quaternion_to_matrix(rotations)
        elif rotations.shape[-1] == 6:
            rot_matrices = rotation_6d_to_matrix(rotations)
        else:
            raise NotImplementedError(f'Unimplemented rotation representation in FK layer, shape of {rotations.shape}')

        rot_matrices = torch.cat([rot_matrices, positions[..., None]], dim=-1)
        zeros = torch.zeros(rot_matrices.shape[:-2] + torch.Size([1, 3])).to(self.device)
        ones = torch.ones(rot_matrices.shape[:-2] + torch.Size([1, 1])).to(self.device)
        zerosones = torch.cat([zeros, ones], dim=-1)
        rot_matrices = torch.cat([rot_matrices, zerosones], dim=-2)
        return rot_matrices

    def rotate_global(self, parents, positions, rotations):
        locals = self.make_fast_rotation_matrices(positions, rotations)
        globals = self.identity_rotation(rotations)

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1)
        b_size = positions.shape[0]
        if self.b_idxs is None:
            self.b_idxs = torch.LongTensor(np.arange(b_size)).to(self.device)
        elif self.b_idxs.shape[-1] != b_size:
            self.b_idxs = torch.LongTensor(np.arange(b_size)).to(self.device)

        for i in range(1, positions.shape[1]):
            globals[:, i] = self.rotate(
                globals[self.b_idxs, parents[i]], locals[:, i])

        return globals

    def get_tpose_joints(self, offsets, parents):
        num_joints = len(parents)
        joints = [offsets[:, 0]]
        for j in range(1, len(parents)):
            joints.append(joints[parents[j]] + offsets[:, j])

        return torch.stack(joints, dim=1)

    def canonical_to_local(self, canonical_xform, global_orient=None):
        """
        Args:
            canonical_xform: (B, J, 3, 3)
            global_orient: (B, 3, 3)

        Returns:
            local_xform: (B, J, 3, 3)
        """
        local_xform = torch.zeros_like(canonical_xform)

        if global_orient is None:
            global_xform = canonical_xform
        else:
            global_xform = torch.matmul(global_orient.unsqueeze(1), canonical_xform)
        for i in range(global_xform.shape[1]):
            if i == 0:
                local_xform[:, i] = global_xform[:, i]
            else:
                local_xform[:, i] = torch.bmm(torch.linalg.inv(global_xform[:, self.parents[i]]), global_xform[:, i])

        return local_xform

    def global_to_local(self, global_xform):
        """
        Args:
            global_xform: (B, J, 3, 3)

        Returns:
            local_xform: (B, J, 3, 3)
        """
        local_xform = torch.zeros_like(global_xform)

        for i in range(global_xform.shape[1]):
            if i == 0:
                local_xform[:, i] = global_xform[:, i]
            else:
                local_xform[:, i] = torch.bmm(torch.linalg.inv(global_xform[:, self.parents[i]]), global_xform[:, i])

        return local_xform

    def forward(self, rotations, positions=None):
        """
        Args:
            rotations (B, J, D)

        Returns:
            The global position of each joint after FK (B, J, 3)
        """
        # Get the full transform with rotations for skinning
        b_size = rotations.shape[0]
        if positions is None:
            positions = self.positions.repeat(b_size, 1, 1)
        transforms = self.rotate_global(self.parents, positions, rotations)
        coordinates = transforms[:, :, :3, 3] / transforms[:, :, 3:, 3]

        return coordinates, transforms
