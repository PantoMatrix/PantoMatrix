import numpy as np
import glob
import os
import pickle
import lmdb
import pyarrow
import fasttext
from loguru import logger
from scipy import linalg
from .pymo.parsers import BVHParser
from .pymo.viz_tools import *
from .pymo.preprocessing import *


joints_list = {
    "trinity_joints":{
        'Hips':         [6,6],
        'Spine':        [3,9],
        'Spine1':       [3,12],
        'Spine2':       [3,15],
        'Spine3':       [3,18],
        'Neck':         [3,21],
        'Neck1':        [3,24],
        'Head':         [3,27],
        'RShoulder':    [3,30], 
        'RArm':         [3,33],
        'RArm1':        [3,36],
        'RHand':        [3,39],
        'RHandT1':      [3,42],
        'RHandT2':      [3,45],
        'RHandT3':      [3,48],
        'RHandI1':      [3,51],
        'RHandI2':      [3,54],
        'RHandI3':      [3,57],
        'RHandM1':      [3,60],
        'RHandM2':      [3,63],
        'RHandM3':      [3,66],
        'RHandR1':      [3,69],
        'RHandR2':      [3,72],
        'RHandR3':      [3,75],
        'RHandP1':      [3,78],
        'RHandP2':      [3,81],
        'RHandP3':      [3,84],
        'LShoulder':    [3,87], 
        'LArm':         [3,90],
        'LArm1':        [3,93],
        'LHand':        [3,96], 
        'LHandT1':      [3,99],
        'LHandT2':      [3,102],
        'LHandT3':      [3,105],
        'LHandI1':      [3,108],
        'LHandI2':      [3,111],
        'LHandI3':      [3,114],
        'LHandM1':      [3,117],
        'LHandM2':      [3,120],
        'LHandM3':      [3,123],
        'LHandR1':      [3,126],
        'LHandR2':      [3,129],
        'LHandR3':      [3,132],
        'LHandP1':      [3,135],
        'LHandP2':      [3,138],
        'LHandP3':      [3,141],
        'RUpLeg':       [3,144],
        'RLeg':         [3,147],
        'RFoot':        [3,150],
        'RFootF':       [3,153],
        'RToeBase':     [3,156],
        'LUpLeg':       [3,159],
        'LLeg':         [3,162],
        'LFoot':        [3,165],
        'LFootF':       [3,168],
        'LToeBase':     [3,171],},
    "trinity_joints_123":{ 
        'Spine':       3 ,
        'Neck':        3 ,
        'Neck1':       3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,
        'RHandT1':     3 ,
        'RHandT2':     3 ,
        'RHandT3':     3 ,
        'RHandI1':     3 ,
        'RHandI2':     3 ,
        'RHandI3':     3 ,
        'RHandM1':     3 ,
        'RHandM2':     3 ,
        'RHandM3':     3 ,
        'RHandR1':     3 ,
        'RHandR2':     3 ,
        'RHandR3':     3 ,
        'RHandP1':     3 ,
        'RHandP2':     3 ,
        'RHandP3':     3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 ,
        'LHandT1':     3 ,
        'LHandT2':     3 ,
        'LHandT3':     3 ,
        'LHandI1':     3 ,
        'LHandI2':     3 ,
        'LHandI3':     3 ,
        'LHandM1':     3 ,
        'LHandM2':     3 ,
        'LHandM3':     3 ,
        'LHandR1':     3 ,
        'LHandR2':     3 ,
        'LHandR3':     3 ,
        'LHandP1':     3 ,
        'LHandP2':     3 ,
        'LHandP3':     3 ,},
    "trinity_joints_168":{
        'Hips':        3 , 
        'Spine':       3 ,
        'Spine1':      3 ,
        'Spine2':      3 ,
        'Spine3':      3 ,
        'Neck':        3 ,
        'Neck1':       3 ,
        'Head':        3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,
        'RHandT1':     3 ,
        'RHandT2':     3 ,
        'RHandT3':     3 ,
        'RHandI1':     3 ,
        'RHandI2':     3 ,
        'RHandI3':     3 ,
        'RHandM1':     3 ,
        'RHandM2':     3 ,
        'RHandM3':     3 ,
        'RHandR1':     3 ,
        'RHandR2':     3 ,
        'RHandR3':     3 ,
        'RHandP1':     3 ,
        'RHandP2':     3 ,
        'RHandP3':     3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 ,
        'LHandT1':     3 ,
        'LHandT2':     3 ,
        'LHandT3':     3 ,
        'LHandI1':     3 ,
        'LHandI2':     3 ,
        'LHandI3':     3 ,
        'LHandM1':     3 ,
        'LHandM2':     3 ,
        'LHandM3':     3 ,
        'LHandR1':     3 ,
        'LHandR2':     3 ,
        'LHandR3':     3 ,
        'LHandP1':     3 ,
        'LHandP2':     3 ,
        'LHandP3':     3 ,
        'RUpLeg':      3 ,
        'RLeg':        3 ,
        'RFoot':       3 ,
        'RFootF':      3 ,
        'RToeBase':    3 ,
        'LUpLeg':      3 ,
        'LLeg':        3 ,
        'LFoot':       3 ,
        'LFootF':      3 ,
        'LToeBase':    3 ,},
    "trinity_joints_138":{
        "Hips":        3 ,
        'Spine':       3 ,
        'Spine1':      3 ,
        'Spine2':      3 ,
        'Spine3':      3 ,
        'Neck':        3 ,
        'Neck1':       3 ,
        'Head':        3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,
        'RHandT1':     3 ,
        'RHandT2':     3 ,
        'RHandT3':     3 ,
        'RHandI1':     3 ,
        'RHandI2':     3 ,
        'RHandI3':     3 ,
        'RHandM1':     3 ,
        'RHandM2':     3 ,
        'RHandM3':     3 ,
        'RHandR1':     3 ,
        'RHandR2':     3 ,
        'RHandR3':     3 ,
        'RHandP1':     3 ,
        'RHandP2':     3 ,
        'RHandP3':     3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 ,
        'LHandT1':     3 ,
        'LHandT2':     3 ,
        'LHandT3':     3 ,
        'LHandI1':     3 ,
        'LHandI2':     3 ,
        'LHandI3':     3 ,
        'LHandM1':     3 ,
        'LHandM2':     3 ,
        'LHandM3':     3 ,
        'LHandR1':     3 ,
        'LHandR2':     3 ,
        'LHandR3':     3 ,
        'LHandP1':     3 ,
        'LHandP2':     3 ,
        'LHandP3':     3 ,},
    
    "beat_joints": {
        'Hips':         [6,6],
        'Spine':        [3,9],
        'Spine1':       [3,12],
        'Spine2':       [3,15],
        'Spine3':       [3,18],
        'Neck':         [3,21],
        'Neck1':        [3,24],
        'Head':         [3,27],
        'HeadEnd':      [3,30],

        'RShoulder':    [3,33], 
        'RArm':         [3,36],
        'RArm1':        [3,39],
        'RHand':        [3,42],    
        'RHandM1':      [3,45],
        'RHandM2':      [3,48],
        'RHandM3':      [3,51],
        'RHandM4':      [3,54],

        'RHandR':       [3,57],
        'RHandR1':      [3,60],
        'RHandR2':      [3,63],
        'RHandR3':      [3,66],
        'RHandR4':      [3,69],

        'RHandP':       [3,72],
        'RHandP1':      [3,75],
        'RHandP2':      [3,78],
        'RHandP3':      [3,81],
        'RHandP4':      [3,84],

        'RHandI':       [3,87],
        'RHandI1':      [3,90],
        'RHandI2':      [3,93],
        'RHandI3':      [3,96],
        'RHandI4':      [3,99],

        'RHandT1':      [3,102],
        'RHandT2':      [3,105],
        'RHandT3':      [3,108],
        'RHandT4':      [3,111],

        'LShoulder':    [3,114], 
        'LArm':         [3,117],
        'LArm1':        [3,120],
        'LHand':        [3,123],    
        'LHandM1':      [3,126],
        'LHandM2':      [3,129],
        'LHandM3':      [3,132],
        'LHandM4':      [3,135],

        'LHandR':       [3,138],
        'LHandR1':      [3,141],
        'LHandR2':      [3,144],
        'LHandR3':      [3,147],
        'LHandR4':      [3,150],

        'LHandP':       [3,153],
        'LHandP1':      [3,156],
        'LHandP2':      [3,159],
        'LHandP3':      [3,162],
        'LHandP4':      [3,165],

        'LHandI':       [3,168],
        'LHandI1':      [3,171],
        'LHandI2':      [3,174],
        'LHandI3':      [3,177],
        'LHandI4':      [3,180],

        'LHandT1':      [3,183],
        'LHandT2':      [3,186],
        'LHandT3':      [3,189],
        'LHandT4':      [3,192],

        'RUpLeg':       [3,195],
        'RLeg':         [3,198],
        'RFoot':        [3,201],
        'RFootF':       [3,204],
        'RToeBase':     [3,207],
        'RToeBaseEnd':  [3,210],

        'LUpLeg':       [3,213],
        'LLeg':         [3,216],
        'LFoot':        [3,219],
        'LFootF':       [3,222],
        'LToeBase':     [3,225],
        'LToeBaseEnd':  [3,228],},
    
    "spine_neck_141":{
        'Spine':       3 ,
        'Neck':        3 ,
        'Neck1':       3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,    
        'RHandM1':     3 ,
        'RHandM2':     3 ,
        'RHandM3':     3 ,
        'RHandR':      3 ,
        'RHandR1':     3 ,
        'RHandR2':     3 ,
        'RHandR3':     3 ,
        'RHandP':      3 ,
        'RHandP1':     3 ,
        'RHandP2':     3 ,
        'RHandP3':     3 ,
        'RHandI':      3 ,
        'RHandI1':     3 ,
        'RHandI2':     3 ,
        'RHandI3':     3 ,
        'RHandT1':     3 ,
        'RHandT2':     3 ,
        'RHandT3':     3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 ,    
        'LHandM1':     3 ,
        'LHandM2':     3 ,
        'LHandM3':     3 ,
        'LHandR':      3 ,
        'LHandR1':     3 ,
        'LHandR2':     3 ,
        'LHandR3':     3 ,
        'LHandP':      3 ,
        'LHandP1':     3 ,
        'LHandP2':     3 ,
        'LHandP3':     3 ,
        'LHandI':      3 ,
        'LHandI1':     3 ,
        'LHandI2':     3 ,
        'LHandI3':     3 ,
        'LHandT1':     3 ,
        'LHandT2':     3 ,
        'LHandT3':     3 ,},
}


class FIDCalculator(object):
    def __init__(self):
        self.gt_rot = None # pandas dataframe for n frames * joints * 6
        self.gt_pos = None # n frames * (joints + 13) * 3
        self.op_rot = None # pandas dataframe for n frames * joints * 6
        self.op_pos = None # n frames * (joints + 13) * 3
        

    def load(self, path, load_type, save_pos=False):
        '''
        select gt or op for load_type
        '''
        parser = BVHParser()
        parsed_data = parser.parse(path)
        if load_type == 'gt':
            self.gt_rot = parsed_data.values
        elif load_type == 'op':
            self.op_rot = parsed_data.values
        else: print('error, select gt or op for load_type')

        if save_pos:
            mp = MocapParameterizer('position')
            positions = mp.fit_transform([parsed_data])
            if load_type == 'gt':
                self.gt_pos = positions[0].values
            elif load_type == 'op':
                self.op_pos = positions[0].values
            else: print('error, select gt or op for load_type')


    def _joint_selector(self, selected_joints, ori_data):
        selected_data = pd.DataFrame(columns=[])

        for joint_name in selected_joints:
            selected_data[joint_name] = ori_data[joint_name]
        return selected_data.to_numpy()
    
    
    def cal_vol(self, dtype):
        if dtype == 'pos':
            gt = self.gt_pos
            op = self.op_pos
        else:
            gt = self.gt_rot
            op = self.op_rot
        
        gt_v = gt.to_numpy()[1:, :] - gt.to_numpy()[0:-1, :]
        op_v = op.to_numpy()[1:, :] - op.to_numpy()[0:-1, :]
        if dtype == 'pos':
            self.gt_vol_pos = pd.DataFrame(gt_v, columns = gt.columns.tolist())
            self.op_vol_pos = pd.DataFrame(op_v, columns = gt.columns.tolist())
        else:
            self.gt_vol_rot = pd.DataFrame(gt_v, columns = gt.columns.tolist())
            self.op_vol_rot = pd.DataFrame(op_v, columns = gt.columns.tolist())


    @staticmethod
    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = FIDCalculator.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist


    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                    inception net (like returned by the function 'get_predictions')
                    for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                    representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                    representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    
    def calculate_fid(self, cal_type, joint_type, high_level_opt):
        
        if cal_type == 'pos':
            if self.gt_pos.shape != self.op_pos.shape:
                min_val = min(self.gt_pos.shape[0],self.op_pos.shape[0])
                gt = self.gt_pos[:min_val]
                op = self.op_pos[:min_val]
            else:
                gt = self.gt_pos
                op = self.op_pos
            full_body = gt.columns.tolist()
        elif cal_type == 'rot':
            if self.gt_rot.shape != self.op_rot.shape:
                min_val = min(self.gt_rot.shape[0],self.op_rot.shape[0])
                gt = self.gt_rot[:min_val]
                op = self.op_rot[:min_val]
            else:
                gt = self.gt_rot
                op = self.op_rot
            full_body_with_offset = gt.columns.tolist()
            full_body = [o for o in full_body_with_offset if ('position' not in o)]
        elif cal_type == 'pos_vol':
            assert self.gt_vol_pos.shape == self.op_vol_pos.shape
            gt = self.gt_vol_pos
            op = self.op_vol_pos
            full_body_with_offset = gt.columns.tolist()
            full_body = gt.columns.tolist()
        elif cal_type == 'rot_vol':
            assert self.gt_vol_rot.shape == self.op_vol_rot.shape
            gt = self.gt_vol_rot
            op = self.op_vol_rot
            full_body_with_offset = gt.columns.tolist()
            full_body = [o for o in full_body_with_offset if ('position' not in o)]       
        #print(f'full_body contains {len(full_body)//3} joints')

        if joint_type == 'full_upper_body':
            selected_body = [o for o in full_body if ('Leg' not in o) and ('Foot' not in o) and ('Toe' not in o)] 
        elif joint_type == 'upper_body':
            selected_body = [o for o in full_body if ('Hand' not in o) and ('Leg' not in o) and ('Foot' not in o) and ('Toe' not in o)]
        elif joint_type == 'fingers':
            selected_body = [o for o in full_body if ('Hand' in o)]
        elif joint_type == 'indivdual':
            pass
        else: print('error, plz select correct joint type')
        #print(f'calculate fid for {len(selected_body)//3} joints')

        gt = self._joint_selector(selected_body, gt)
        op = self._joint_selector(selected_body, op)

        if high_level_opt == 'fid':
            fid = FIDCalculator.frechet_distance(gt, op)
            return fid
        elif high_level_opt == 'var':
            var_gt = gt.var()
            var_op = op.var()
            return var_gt, var_op
        elif high_level_opt == 'mean':
            mean_gt = gt.mean()
            mean_op = op.mean()
            return mean_gt, mean_op
        else: return 0
             

def result2target_vis(pose_version, res_bvhlist, save_path, demo_name, verbose=True):
    if "trinity" in pose_version:
        ori_list = joints_list[pose_version[6:-4]] 
        target_list = joints_list[pose_version[6:]] 
        file_content_length = 336 
    elif "beat" in pose_version or "spine_neck_141" in pose_version:
        ori_list = joints_list["beat_joints"]
        target_list = joints_list["spine_neck_141"]
        file_content_length = 431
    else:
        pass
    
    bvh_files_dirs = sorted(glob.glob(f'{res_bvhlist}*.bvh'), key=str)
    counter = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, bvh_file_dir in enumerate(bvh_files_dirs):
        short_name = bvh_file_dir.split("/")[-1][11:]
        #print(short_name)
        wirte_file =  open(os.path.join(save_path, f'res_{short_name}'),'w+')
        with open(f"{demo_name}{short_name}",'r') as pose_data_pre:
            pose_data_pre_file = pose_data_pre.readlines()
            for j, line in enumerate(pose_data_pre_file[0:file_content_length]):
                    wirte_file.write(line)
            offset_data = pose_data_pre_file[file_content_length]
            offset_data = np.fromstring(offset_data, dtype=float, sep=' ')
        wirte_file.close()

        wirte_file = open(os.path.join(save_path, f'res_{short_name}'),'r')
        ori_lines = wirte_file.readlines()
        with open(bvh_file_dir, 'r') as pose_data:
            pose_data_file = pose_data.readlines()
        ori_lines[file_content_length-2] = 'Frames: ' + str(len(pose_data_file)-1) + '\n'
        wirte_file.close() 

        wirte_file = open(os.path.join(save_path, f'res_{short_name}'),'w+')
        wirte_file.writelines(i for i in ori_lines[:file_content_length])    
        wirte_file.close() 

        with open(os.path.join(save_path, f'res_{short_name}'),'a+') as wirte_file: 
            with open(bvh_file_dir, 'r') as pose_data:
                data_each_file = []
                pose_data_file = pose_data.readlines()
                for j, line in enumerate(pose_data_file):
                    if not j:
                        pass
                    else:          
                        data = np.fromstring(line, dtype=float, sep=' ')
                        data_rotation = offset_data.copy()   
                        for iii, (k, v) in enumerate(target_list.items()): # here is 147 rotations by 3
                            #print(data_rotation[ori_list[k][1]-v:ori_list[k][1]], data[iii*3:iii*3+3])
                            data_rotation[ori_list[k][1]-v:ori_list[k][1]] = data[iii*3:iii*3+3]
                        data_each_file.append(data_rotation)
        
            for line_data in data_each_file:
                line_data = np.array2string(line_data, max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                wirte_file.write(line_data[1:-2]+'\n')

        counter += 1
        if verbose:
            logger.info('data_shape:', data_rotation.shape, 'process:', counter, '/', len(bvh_files_dirs))