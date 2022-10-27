import glob
from bvh import Bvh
import math
import numpy as np
import collections
import os

beat_joints = collections.OrderedDict()

beat_joints={
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
        'LToeBaseEnd':  [3,228],}


joint_name_list_186 =  {
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
        'LHandT3':     3 ,
        'RUpLeg':      3 ,
        'RLeg':        3 ,
        'RFoot':       3 ,
        'RFootF':      3 ,
        'RToeBase':    3 ,
        'LUpLeg':      3 ,
        'LLeg':        3 ,
        'LFoot':       3 ,
        'LFootF':      3 ,
        'LToeBase':    3 ,}


joint_name_list_27 =  {
        'Hips':        3 , 
        'Neck':        3 ,
        'Head':        3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,    
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 , }   


spine_neck_141 =  { 
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
        'LHandT3':     3 ,}


joint_name_list_27_v3 =  {
        'Spine3':      3 , 
        'Neck':        3 ,
        'Head':        3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,    
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 , }  
     

def get_mean_pose(bvh_path):
    bvh_files_dirs = sorted(glob.glob(f'{bvh_path}/*.bvh'))
    data_all_file = []
    for i, bvh_files_dir in enumerate(bvh_files_dirs):
        with open(bvh_files_dir, 'r') as pose_data:
            for j, line in enumerate(pose_data.readlines()):
                    data = np.fromstring(line, dtype=float, sep=' ')
                    data_all_file.append(data) 
                    print(i, '/', len(bvh_files_dirs), ' ', j-1)
    data_all_file = np.array(data_all_file) 
    
    # print(data_all_file.shape)
    mean_pose = np.mean(data_all_file, axis=0)
    std_pose = np.std(data_all_file, axis=0)
    print(std_pose)
    count = 0
    for i in range(len(std_pose)//3):
        count +=1
        max_val = max(std_pose[i*3:i*3+2])
        min_val = min(std_pose[i*3:i*3+2])
        print('max:',max_val,'min:',min_val )
        if min_val < 1:
            print(count)
    with open(f'{bvh_path}/mean.npy', 'wb') as f:
        np.save(f, mean_pose)
    with open(f'{bvh_path}/std.npy', 'wb') as f:
        #print()
        np.save(f, std_pose)


def list_check(joint_name_list):
    count_joints = 0
    count_rotations = 0
    for _, j_numbers in joint_name_list.items():
        count_joints += 1
        count_rotations += j_numbers[0]
        # print(count_rotations)
    print('joints:', count_joints, 'freedom:', count_rotations)
    return count_joints, count_rotations


def transfer2target(ori_list, target_list, ori_path, save_path, target_fps):
    bvh_files_dirs = sorted(glob.glob(f'{ori_path}*.bvh'))
    counter = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for bvh_file_dir in bvh_files_dirs:
        with open(bvh_file_dir, 'r') as pose_data:
            data_each_file = []
            for j, line in enumerate(pose_data.readlines()[430:]):
               
                if not j:
                    words = line.split()
                    FPS = math.ceil(1/float(words[-1]))
                    factor = math.ceil(FPS/target_fps)
                    print('original FPS:', FPS, 'reduce factor:', factor)
                else:          
                    if j % factor != 1 and factor != 1:
                        continue
                    data = np.fromstring(line, dtype=float, sep=' ')
                    data_rotation = np.zeros((1))   
                    for k, v in target_list.items():

                        data_rotation = np.concatenate((data_rotation, data[ori_list[k][1]-v:ori_list[k][1]]))
                    data_each_file.append(data_rotation[1:])
            

        with open(os.path.join(save_path, f'fps{target_fps}_{bvh_file_dir[-11:]}'),'w+') as wirte_file:
            for line_data in data_each_file:
                line_data = np.array2string(line_data, max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                wirte_file.write(line_data[1:-2]+'\n')

        counter += 1
        print('data_shape:', data_rotation[1:].shape, 'process:', counter, '/', len(bvh_files_dirs)) 


def transfer2target_vis(ori_list, target_list, ori_path, save_path, target_fps):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bvh_files_dirs = sorted(glob.glob(f'{ori_path}*.bvh'))
    with open('/home/ma-user/work/datasets/beat/train/bvh30fps/anger_cut_32_43_002_074.bvh', 'r') as pose_data:
        with open(os.path.join(save_path, 'demo.bvh'),'w+') as wirte_file:
            pose_data_file = pose_data.readlines()
            for j, line in enumerate(pose_data_file[0:429]):
                    wirte_file.write(line)
            for j, line in enumerate(pose_data_file[431:432]):
                    first_line = line
            offset_data = np.fromstring(first_line, dtype=float, sep=' ')
            for i in range(len(offset_data)//6):
                offset_data[i*6+3] = 0
                offset_data[i*6+4] = 0
                offset_data[i*6+5] = 0 
    
    counter = 0
    for bvh_file_dir in bvh_files_dirs:
        with open(os.path.join(save_path, f'fps{target_fps}_{bvh_file_dir[-11:]}'),'w+') as wirte_file:
            with open(os.path.join(save_path, 'demo.bvh'),'r') as pose_data_pre:
                for j, line in enumerate(pose_data_pre.readlines()[0:429]):
                        wirte_file.write(line)

            with open(bvh_file_dir, 'r') as pose_data:
                data_each_file = []
                # Frames: xxxx
                pose_data_file = pose_data.readlines()
                for j, line in enumerate(pose_data_file[430:]):
                
                    if not j:
                        words = line.split()
                        FPS = math.ceil(1/float(words[-1]))
                        factor = math.ceil(FPS/target_fps)
                        frames = int(pose_data_file[429][7:])//factor
                        wirte_file.write(pose_data_file[429][:8] + str(frames) +'\n')
                        wirte_file.write(pose_data_file[430][:12] + str(float(words[-1]) * 4) +'\n')
                        print('original FPS:', FPS, 'reduce factor:', factor, 'frames:', frames)
                    else:          
                        if j % factor != 1 and factor != 1:
                            continue
                        data = np.fromstring(line, dtype=float, sep=' ')
                        data_rotation = offset_data.copy()   
                        for k, v in target_list.items():
                            data_rotation[ori_list[k][1]-v:ori_list[k][1]] = data[ori_list[k][1]-v:ori_list[k][1]]
                            # data_rotation = np.concatenate((data_rotation, data[ori_list[k][1]-v:ori_list[k][1]]))
                        data_each_file.append(data_rotation)
        
            for line_data in data_each_file:
                line_data = np.array2string(line_data, max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                wirte_file.write(line_data[1:-2]+'\n')

        counter += 1
        print('data_shape:', data_rotation.shape, 'process:', counter, '/', len(bvh_files_dirs))


if __name__ == "__main__":
    #list_check(joint_name_list)
    transfer2target(beat_joints, spine_neck_141, '/home/ma-user/work/datasets/beat/train/bvh30fps/', '/home/ma-user/work/datasets/beat/train/toy_data_beat/', 15)
    transfer2target(beat_joints, spine_neck_141, '/home/ma-user/work/datasets/beat/val/bvh30fps/', '/home/ma-user/work/datasets/beat/val/toy_data_beat/', 15)
    transfer2target(beat_joints, spine_neck_141, '/home/ma-user/work/datasets/beat/test/bvh30fps/', '/home/ma-user/work/datasets/beat/test/toy_data_beat/', 15)
    get_mean_pose('/home/ma-user/work/datasets/beat/train/toy_data_beat/')
    
    transfer2target_vis(beat_joints, spine_neck_141, '/home/ma-user/work/datasets/beat/train/bvh30fps/', '/home/ma-user/work/datasets/beat/train/toy_data_beat_vis/', 15)
    transfer2target_vis(beat_joints, spine_neck_141, '/home/ma-user/work/datasets/beat/val/bvh30fps/', '/home/ma-user/work/datasets/beat/val/toy_data_beat_vis/', 15)
    transfer2target_vis(beat_joints, spine_neck_141, '/home/ma-user/work/datasets/beat/test/bvh30fps/', '/home/ma-user/work/datasets/beat/test/toy_data_beat_vis/',15)
