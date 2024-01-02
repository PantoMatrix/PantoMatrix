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




# pose version fpsxx_trinity/japanese_joints(_xxx)
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
    "beat_smplx_joints": {
        'pelvis':         [3,3],
        'left_hip':        [3,6],
        'right_hip':       [3,9],
        'spine1':       [3,12],
        'left_knee':       [3,15],
        'right_knee':         [3,18],
        'spine2':        [3,21],
        'left_ankle':         [3,24],
        'right_ankle':      [3,27],

        'spine3':    [3,30], 
        'left_foot':         [3,33],
        'right_foot':        [3,36],
        'neck':        [3,39],    
        'left_collar':      [3,42],
        'right_collar':      [3,45],
        'head':      [3,48],
        'left_shoulder':      [3,51],

        'right_shoulder':       [3,54],
        'left_elbow':      [3,57],
        'right_elbow':      [3,60],
        'left_wrist':      [3,63],
        'right_wrist':      [3,66],

        'jaw':       [3,69],
        'left_eye_smplhf':      [3,72],
        'right_eye_smplhf':      [3,75],
        'left_index1':      [3,78],
        'left_index2':      [3,81],

        'left_index3':       [3,84],
        'left_middle1':      [3,87],
        'left_middle2':      [3,90],
        'left_middle3':      [3,93],
        'left_pinky1':      [3,96],

        'left_pinky2':      [3,99],
        'left_pinky3':      [3,102],
        'left_ring1':      [3,105],
        'left_ring2':      [3,108],

        'left_ring3':    [3,111], 
        'left_thumb1':         [3,114],
        'left_thumb2':        [3,117],
        'left_thumb3':        [3,120],    
        'right_index1':      [3,123],
        'right_index2':      [3,126],
        'right_index3':      [3,129],
        'right_middle1':      [3,132],

        'right_middle2':       [3,135],
        'right_middle3':      [3,138],
        'right_pinky1':      [3,141],
        'right_pinky2':      [3,144],
        'right_pinky3':      [3,147],

        'right_ring1':       [3,150],
        'right_ring2':      [3,153],
        'right_ring3':      [3,156],
        'right_thumb1':      [3,159],
        'right_thumb2':      [3,162],
        'right_thumb3':       [3,165],
        
#         'nose':      [3,168],
#         'right_eye':      [3,171],
#         'left_eye':      [3,174],
#         'right_ear':      [3,177],

#         'left_ear':      [3,180],
#         'left_big_toe':      [3,183],
#         'left_small_toe':      [3,186],
#         'left_heel':      [3,189],

#         'right_big_toe':       [3,192],
#         'right_small_toe':         [3,195],
#         'right_heel':        [3,198],
#         'left_thumb':       [3,201],
#         'left_index':     [3,204],
#         'left_middle':  [3,207],

#         'left_ring':       [3,210],
#         'left_pinky':         [3,213],
#         'right_thumb':        [3,216],
#         'right_index':       [3,219],
#         'right_middle':     [3,222],
#         'right_ring':  [3,225],
        
#         'right_pinky':      [3,228],
#         'right_eye_brow1':      [3,231],
#         'right_eye_brow2':      [3,234],
#         'right_eye_brow3':      [3,237],

#         'right_eye_brow4':      [3,240],
#         'right_eye_brow5':      [3,243],
#         'left_eye_brow5':      [3,246],
#         'left_eye_brow4':      [3,249],

#         'left_eye_brow3':       [3,252],
#         'left_eye_brow2':         [3,255],
#         'left_eye_brow1':        [3,258],
#         'nose1':       [3,261],
#         'nose2':     [3,264],
#         'nose3':  [3,267],

#         'nose4':       [3,270],
#         'right_nose_2':         [3,273],
#         'right_nose_1':        [3,276],
#         'nose_middle':       [3,279],
#         'left_nose_1':     [3,282],
#         'left_nose_2':  [3,285],
        
#         'right_eye1':      [3,288],
#         'right_eye2':      [3,291],
#         'right_eye3':      [3,294],
#         'right_eye4':      [3,297],

#         'right_eye5':      [3,300],
#         'right_eye6':      [3,303],
#         'left_eye4':      [3,306],
#         'left_eye3':      [3,309],

#         'left_eye2':       [3,312],
#         'left_eye1':         [3,315],
#         'left_eye6':        [3,318],
#         'left_eye5':       [3,321],
#         'right_mouth_1':     [3,324],
#         'right_mouth_2':  [3,327],
#         'right_mouth_3':       [3,330],
#         'mouth_top':         [3,333],
#         'left_mouth_3':        [3,336],
#         'left_mouth_2':       [3,339],
#         'left_mouth_1':     [3,342],
#         'left_mouth_5':  [3,345],
#         'left_mouth_4':        [3,348],
#         'mouth_bottom':       [3,351],
#         'right_mouth_4':     [3,354],
#         'right_mouth_5':  [3,357],
#         'right_lip_1':        [3,360],
#         'right_lip_2':       [3,363],
#         'lip_top':     [3,366],
#         'left_lip_2':  [3,369],
        
#         'left_lip_1':       [3,372],
#         'left_lip_3':         [3,375],
#         'lip_bottom':        [3,378],
#         'right_lip_3':       [3,381],
#         'right_contour_1':     [3,384],
#         'right_contour_2':  [3,387],
#         'right_contour_3':       [3,390],
#         'right_contour_4':         [3,393],
#         'right_contour_5':        [3,396],
#         'right_contour_6':       [3,399],
#         'right_contour_7':     [3,402],
#         'right_contour_8':  [3,405],
#         'contour_middle':        [3,408],
#         'left_contour_8':       [3,411],
#         'left_contour_7':     [3,414],
#         'left_contour_6':  [3,417],
#         'left_contour_5':        [3,420],
#         'left_contour_4':       [3,423],
#         'left_contour_3':     [3,426],
#         'left_contour_2':  [3,429],
#         'left_contour_1':  [3,432],
    },
    
    "beat_smplx_no_eyes": {
        "pelvis":3,
        "left_hip":3,
        "right_hip":3,
        "spine1":3,
        "left_knee":3,
        "right_knee":3,
        "spine2":3,
        "left_ankle":3,
        "right_ankle":3,
        "spine3":3,
        "left_foot":3,
        "right_foot":3,
        "neck":3,
        "left_collar":3,
        "right_collar":3,
        "head":3,
        "left_shoulder":3,
        "right_shoulder":3,
        "left_elbow":3,
        "right_elbow":3,
        "left_wrist":3,
        "right_wrist":3,
        "jaw":3,
        # "left_eye_smplhf":3,
        # "right_eye_smplhf":3,
        "left_index1":3,
        "left_index2":3,
        "left_index3":3,
        "left_middle1":3,
        "left_middle2":3,
        "left_middle3":3,
        "left_pinky1":3,
        "left_pinky2":3,
        "left_pinky3":3,
        "left_ring1":3,
        "left_ring2":3,
        "left_ring3":3,
        "left_thumb1":3,
        "left_thumb2":3,
        "left_thumb3":3,
        "right_index1":3,
        "right_index2":3,
        "right_index3":3,
        "right_middle1":3,
        "right_middle2":3,
        "right_middle3":3,
        "right_pinky1":3,
        "right_pinky2":3,
        "right_pinky3":3,
        "right_ring1":3,
        "right_ring2":3,
        "right_ring3":3,
        "right_thumb1":3,
        "right_thumb2":3,
        "right_thumb3":3,
    },
    
    "beat_smplx_full": {
        "pelvis":3,
        "left_hip":3,
        "right_hip":3,
        "spine1":3,
        "left_knee":3,
        "right_knee":3,
        "spine2":3,
        "left_ankle":3,
        "right_ankle":3,
        "spine3":3,
        "left_foot":3,
        "right_foot":3,
        "neck":3,
        "left_collar":3,
        "right_collar":3,
        "head":3,
        "left_shoulder":3,
        "right_shoulder":3,
        "left_elbow":3,
        "right_elbow":3,
        "left_wrist":3,
        "right_wrist":3,
        "jaw":3,
        "left_eye_smplhf":3,
        "right_eye_smplhf":3,
        "left_index1":3,
        "left_index2":3,
        "left_index3":3,
        "left_middle1":3,
        "left_middle2":3,
        "left_middle3":3,
        "left_pinky1":3,
        "left_pinky2":3,
        "left_pinky3":3,
        "left_ring1":3,
        "left_ring2":3,
        "left_ring3":3,
        "left_thumb1":3,
        "left_thumb2":3,
        "left_thumb3":3,
        "right_index1":3,
        "right_index2":3,
        "right_index3":3,
        "right_middle1":3,
        "right_middle2":3,
        "right_middle3":3,
        "right_pinky1":3,
        "right_pinky2":3,
        "right_pinky3":3,
        "right_ring1":3,
        "right_ring2":3,
        "right_ring3":3,
        "right_thumb1":3,
        "right_thumb2":3,
        "right_thumb3":3,
    },

    "beat_smplx_upall": {
        # "pelvis":3,
        # "left_hip":3,
        # "right_hip":3,
        "spine1":3,
        # "left_knee":3,
        # "right_knee":3,
        "spine2":3,
        # "left_ankle":3,
        # "right_ankle":3,
        "spine3":3,
        # "left_foot":3,
        # "right_foot":3,
        "neck":3,
        "left_collar":3,
        "right_collar":3,
        "head":3,
        "left_shoulder":3,
        "right_shoulder":3,
        "left_elbow":3,
        "right_elbow":3,
        "left_wrist":3,
        "right_wrist":3,
        # "jaw":3,
        # "left_eye_smplhf":3,
        # "right_eye_smplhf":3,
        "left_index1":3,
        "left_index2":3,
        "left_index3":3,
        "left_middle1":3,
        "left_middle2":3,
        "left_middle3":3,
        "left_pinky1":3,
        "left_pinky2":3,
        "left_pinky3":3,
        "left_ring1":3,
        "left_ring2":3,
        "left_ring3":3,
        "left_thumb1":3,
        "left_thumb2":3,
        "left_thumb3":3,
        "right_index1":3,
        "right_index2":3,
        "right_index3":3,
        "right_middle1":3,
        "right_middle2":3,
        "right_middle3":3,
        "right_pinky1":3,
        "right_pinky2":3,
        "right_pinky3":3,
        "right_ring1":3,
        "right_ring2":3,
        "right_ring3":3,
        "right_thumb1":3,
        "right_thumb2":3,
        "right_thumb3":3,
    },

    "beat_smplx_upper": {
        #"pelvis":3,
        # "left_hip":3,
        # "right_hip":3,
        "spine1":3,
        # "left_knee":3,
        # "right_knee":3,
        "spine2":3,
        # "left_ankle":3,
        # "right_ankle":3,
        "spine3":3,
        # "left_foot":3,
        # "right_foot":3,
        "neck":3,
        "left_collar":3,
        "right_collar":3,
        "head":3,
        "left_shoulder":3,
        "right_shoulder":3,
        "left_elbow":3,
        "right_elbow":3,
        "left_wrist":3,
        "right_wrist":3,
        # "jaw":3,
        # "left_eye_smplhf":3,
        # "right_eye_smplhf":3,
        # "left_index1":3,
        # "left_index2":3,
        # "left_index3":3,
        # "left_middle1":3,
        # "left_middle2":3,
        # "left_middle3":3,
        # "left_pinky1":3,
        # "left_pinky2":3,
        # "left_pinky3":3,
        # "left_ring1":3,
        # "left_ring2":3,
        # "left_ring3":3,
        # "left_thumb1":3,
        # "left_thumb2":3,
        # "left_thumb3":3,
        # "right_index1":3,
        # "right_index2":3,
        # "right_index3":3,
        # "right_middle1":3,
        # "right_middle2":3,
        # "right_middle3":3,
        # "right_pinky1":3,
        # "right_pinky2":3,
        # "right_pinky3":3,
        # "right_ring1":3,
        # "right_ring2":3,
        # "right_ring3":3,
        # "right_thumb1":3,
        # "right_thumb2":3,
        # "right_thumb3":3,
    },

        "beat_smplx_hands": {
        #"pelvis":3,
        # "left_hip":3,
        # "right_hip":3,
        # "spine1":3,
        # "left_knee":3,
        # "right_knee":3,
        # "spine2":3,
        # "left_ankle":3,
        # "right_ankle":3,
        # "spine3":3,
        # "left_foot":3,
        # "right_foot":3,
        # "neck":3,
        # "left_collar":3,
        # "right_collar":3,
        # "head":3,
        # "left_shoulder":3,
        # "right_shoulder":3,
        # "left_elbow":3,
        # "right_elbow":3,
        # "left_wrist":3,
        # "right_wrist":3,
        # "jaw":3,
        # "left_eye_smplhf":3,
        # "right_eye_smplhf":3,
        "left_index1":3,
        "left_index2":3,
        "left_index3":3,
        "left_middle1":3,
        "left_middle2":3,
        "left_middle3":3,
        "left_pinky1":3,
        "left_pinky2":3,
        "left_pinky3":3,
        "left_ring1":3,
        "left_ring2":3,
        "left_ring3":3,
        "left_thumb1":3,
        "left_thumb2":3,
        "left_thumb3":3,
        "right_index1":3,
        "right_index2":3,
        "right_index3":3,
        "right_middle1":3,
        "right_middle2":3,
        "right_middle3":3,
        "right_pinky1":3,
        "right_pinky2":3,
        "right_pinky3":3,
        "right_ring1":3,
        "right_ring2":3,
        "right_ring3":3,
        "right_thumb1":3,
        "right_thumb2":3,
        "right_thumb3":3,
    },

    "beat_smplx_lower": {
        "pelvis":3,
        "left_hip":3,
        "right_hip":3,
        # "spine1":3,
        "left_knee":3,
        "right_knee":3,
        # "spine2":3,
        "left_ankle":3,
        "right_ankle":3,
        # "spine3":3,
        "left_foot":3,
        "right_foot":3,
        # "neck":3,
        # "left_collar":3,
        # "right_collar":3,
        # "head":3,
        # "left_shoulder":3,
        # "right_shoulder":3,
        # "left_elbow":3,
        # "right_elbow":3,
        # "left_wrist":3,
        # "right_wrist":3,
        # "jaw":3,
        # "left_eye_smplhf":3,
        # "right_eye_smplhf":3,
        # "left_index1":3,
        # "left_index2":3,
        # "left_index3":3,
        # "left_middle1":3,
        # "left_middle2":3,
        # "left_middle3":3,
        # "left_pinky1":3,
        # "left_pinky2":3,
        # "left_pinky3":3,
        # "left_ring1":3,
        # "left_ring2":3,
        # "left_ring3":3,
        # "left_thumb1":3,
        # "left_thumb2":3,
        # "left_thumb3":3,
        # "right_index1":3,
        # "right_index2":3,
        # "right_index3":3,
        # "right_middle1":3,
        # "right_middle2":3,
        # "right_middle3":3,
        # "right_pinky1":3,
        # "right_pinky2":3,
        # "right_pinky3":3,
        # "right_ring1":3,
        # "right_ring2":3,
        # "right_ring3":3,
        # "right_thumb1":3,
        # "right_thumb2":3,
        # "right_thumb3":3,
    },

    "beat_smplx_face": {
        # "pelvis":3,
        # "left_hip":3,
        # "right_hip":3,
        # # "spine1":3,
        # "left_knee":3,
        # "right_knee":3,
        # # "spine2":3,
        # "left_ankle":3,
        # "right_ankle":3,
        # # "spine3":3,
        # "left_foot":3,
        # "right_foot":3,
        # "neck":3,
        # "left_collar":3,
        # "right_collar":3,
        # "head":3,
        # "left_shoulder":3,
        # "right_shoulder":3,
        # "left_elbow":3,
        # "right_elbow":3,
        # "left_wrist":3,
        # "right_wrist":3,
        "jaw":3,
        # "left_eye_smplhf":3,
        # "right_eye_smplhf":3,
        # "left_index1":3,
        # "left_index2":3,
        # "left_index3":3,
        # "left_middle1":3,
        # "left_middle2":3,
        # "left_middle3":3,
        # "left_pinky1":3,
        # "left_pinky2":3,
        # "left_pinky3":3,
        # "left_ring1":3,
        # "left_ring2":3,
        # "left_ring3":3,
        # "left_thumb1":3,
        # "left_thumb2":3,
        # "left_thumb3":3,
        # "right_index1":3,
        # "right_index2":3,
        # "right_index3":3,
        # "right_middle1":3,
        # "right_middle2":3,
        # "right_middle3":3,
        # "right_pinky1":3,
        # "right_pinky2":3,
        # "right_pinky3":3,
        # "right_ring1":3,
        # "right_ring2":3,
        # "right_ring3":3,
        # "right_thumb1":3,
        # "right_thumb2":3,
        # "right_thumb3":3,
    },
    
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
    
    "beat_full":{
        'Hips': 3,
        'Spine':       3 ,
        'Spine1':       3 ,
        'Spine2':       3 ,
        'Spine3':       3 ,
        'Neck':        3 ,
        'Neck1':       3 ,
        'Head' :       3,
        'HeadEnd' :       3,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,    
        'RHandM1':     3 ,
        'RHandM2':     3 ,
        'RHandM3':     3 ,
        'RHandM4':     3 ,
        'RHandR':      3 ,
        'RHandR1':     3 ,
        'RHandR2':     3 ,
        'RHandR3':     3 ,
        'RHandR4':     3 ,
        'RHandP':      3 ,
        'RHandP1':     3 ,
        'RHandP2':     3 ,
        'RHandP3':     3 ,
        'RHandP4':     3 ,
        'RHandI':      3 ,
        'RHandI1':     3 ,
        'RHandI2':     3 ,
        'RHandI3':     3 ,
        'RHandI4':     3 ,
        'RHandT1':     3 ,
        'RHandT2':     3 ,
        'RHandT3':     3 ,
        'RHandT4':     3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 ,    
        'LHandM1':     3 ,
        'LHandM2':     3 ,
        'LHandM3':     3 ,
        'LHandM4':     3 ,
        'LHandR':      3 ,
        'LHandR1':     3 ,
        'LHandR2':     3 ,
        'LHandR3':     3 ,
        'LHandR4':     3 ,
        'LHandP':      3 ,
        'LHandP1':     3 ,
        'LHandP2':     3 ,
        'LHandP3':     3 ,
        'LHandP4':     3 ,
        'LHandI':      3 ,
        'LHandI1':     3 ,
        'LHandI2':     3 ,
        'LHandI3':     3 ,
        'LHandI4':     3 ,
        'LHandT1':     3 ,
        'LHandT2':     3 ,
        'LHandT3':     3 ,
        'LHandT4':     3 ,
        'RUpLeg':      3,
        'RLeg':        3,
        'RFoot':       3,
        'RFootF':       3,
        'RToeBase':     3,
        'RToeBaseEnd':  3,
        'LUpLeg':       3,
        'LLeg':         3,
        'LFoot':        3,
        'LFootF':       3,
        'LToeBase':     3,
        'LToeBaseEnd':  3,
    },
    
    "japanese_joints":{
        'Hips':         [6,6],
        'Spine':        [6,12],
        'Spine1':       [6,18],
        'Spine2':       [6,24],
        'Spine3':       [6,30],
        'Neck':         [6,36],
        'Neck1':        [6,42],
        'Head':         [6,48],
        'RShoulder':    [6,54], 
        'RArm':         [6,60],
        'RArm1':        [6,66],
        'RHand':        [6,72],    
        'RHandM1':      [6,78],
        'RHandM2':      [6,84],
        'RHandM3':      [6,90],
        'RHandR':       [6,96],
        'RHandR1':      [6,102],
        'RHandR2':      [6,108],
        'RHandR3':      [6,114],
        'RHandP':       [6,120],
        'RHandP1':      [6,126],
        'RHandP2':      [6,132],
        'RHandP3':      [6,138],
        'RHandI':       [6,144],
        'RHandI1':      [6,150],
        'RHandI2':      [6,156],
        'RHandI3':      [6,162],
        'RHandT1':      [6,168],
        'RHandT2':      [6,174],
        'RHandT3':      [6,180],
        'LShoulder':    [6,186], 
        'LArm':         [6,192],
        'LArm1':        [6,198],
        'LHand':        [6,204],    
        'LHandM1':      [6,210],
        'LHandM2':      [6,216],
        'LHandM3':      [6,222],
        'LHandR':       [6,228],
        'LHandR1':      [6,234],
        'LHandR2':      [6,240],
        'LHandR3':      [6,246],
        'LHandP':       [6,252],
        'LHandP1':      [6,258],
        'LHandP2':      [6,264],
        'LHandP3':      [6,270],
        'LHandI':       [6,276],
        'LHandI1':      [6,282],
        'LHandI2':      [6,288],
        'LHandI3':      [6,294],
        'LHandT1':      [6,300],
        'LHandT2':      [6,306],
        'LHandT3':      [6,312],
        'RUpLeg':       [6,318],
        'RLeg':         [6,324],
        'RFoot':        [6,330],
        'RFootF':       [6,336],
        'RToeBase':     [6,342],
        'LUpLeg':       [6,348],
        'LLeg':         [6,354],
        'LFoot':        [6,360],
        'LFootF':       [6,366],
        'LToeBase':     [6,372],},
    
    "yostar":{
    'Hips':         [6,6],
    'Spine':        [3,9],
    'Spine1':       [3,12],
    'Bone040':       [3,15],
    'Bone041':       [3,18],

    'Bone034':         [3,21],
    'Bone035':        [3,24],
    'Bone036':         [3,27],
    'Bone037':        [3,30],
    'Bone038':         [3,33],
    'Bone039':        [3,36],

    'RibbonL1':         [3,39],
    'RibbonL1_end':      [3,42],

    'Chest':         [3,45],
    'L_eri':      [3,48],
    'R_eri':      [3,51],
    'Neck':      [3,54],
    'Head':      [3,57],
    'Head_end':      [3,60],
    
    'RBackHair_1':  [3,63],
    'RBackHair_2':  [3,66],
    'RBackHair_3':  [3,69],
    'RBackHair_4':  [3,72],
    'RBackHair_end':  [3,75],
    
    'RFrontHair':  [3,78],
    'CFrontHair_1':  [3,81],
    'CFrontHair_2':  [3,84],
    'CFrontHair_3':  [3,87],
    'CFrontHair_emd':  [3,90],
    
    'LFrontHair_1':  [3,93],
    'LFrontHair_2':  [3,96],
    'LFrontHair_3':  [3,99],
    
    'LBackHair_1':  [3,102],
    'LBackHair_2':  [3,105],
    'LBackHair_3':  [3,108],
    'LBackHair_4':  [3,111],
    'LBackHair_end':  [3,114],
    
    'LSideHair_1':  [3,117],
    'LSideHair_2':  [3,120],
    'LSideHair_3':  [3,123],
    'LSideHair_4':  [3,126],
    'LSideHair_5':  [3,129],
    'LSideHair_6':  [3,132],
    'LSideHair_7':  [3,135],
    'LSideHair_end':  [3,138],
    
    'CBackHair_1':  [3,141],
    'CBackHair_2':  [3,144],
    'CBackHair_3':  [3,147],
    'CBackHair_4':  [3,150],
    'CBackHair_end':  [3,153],
    
    'RSideHair_1':  [3,156],
    'RSideHair_2':  [3,159],
    'RSideHair_3':  [3,162],
    'RSideHair_4':  [3,165],
    
    'RibbonR_1':  [3,168],
    'RibbonR_2':  [3,171],
    'RibbonR_3':  [3,174],
    
    'RibbonL_1':  [3,177],
    'RibbonL_2':  [3,180],
    'RibbonL_3':  [3,183],
    
    'LeftEye':  [3,186],
    'LeftEye_end':  [3,189],
    'RightEye':  [3,192],
    'RightEye_end':  [3,195],
    
    'LeftShoulder':    [3,198], 
    'LeftArm':         [3,201],
    'LeftForearm':        [3,204],
    'LeftHand':        [3,207],
    'LeftHandThumb1':      [3,210],
    'LeftHandThumb2':      [3,213],
    'LeftHandThumb3':      [3,216],
    'LeftHandThumb_end':      [3,219],
    
    'LeftHandIndex1':       [3,222],
    'LeftHandIndex2':      [3,225],
    'LeftHandIndex3':      [3,228],
    'LeftHandIndex_end':      [3,231],
    
    'LeftHandMiddle1':      [3,234],
    'LeftHandMiddle2':      [3,237],
    'LeftHandMiddle3':      [3,240],
    'LeftHandMiddle_end':      [3,243],

    'LeftHandRing1':       [3,246],
    'LeftHandRing2':      [3,249],
    'LeftHandRing3':      [3,252],
    'LeftHandRing_end':      [3,255],

    'LeftHandPinky1':       [3,258],
    'LeftHandPinky2':      [3,261],
    'LeftHandPinky3':      [3,264],
    'LeftHandPinky_end':      [3,267],

    'RightShoulder':    [3,270], 
    'RightArm':         [3,273],
    'RightForearm':        [3,276],
    'RightHand':        [3,279],
    'RightHandThumb1':      [3,282],
    'RightHandThumb2':      [3,285],
    'RightHandThumb3':      [3,288],
    'RightHandThumb_end':      [3,291],
    
    'RightHandIndex1':       [3,294],
    'RightHandIndex2':      [3,297],
    'RightHandIndex3':      [3,300],
    'RightHandIndex_end':      [3,303],
    
    'RightHandMiddle1':      [3,306],
    'RightHandMiddle2':      [3,309],
    'RightHandMiddle3':      [3,312],
    'RightHandMiddle_end':      [3,315],

    'RightHandRing1':       [3,318],
    'RightHandRing2':      [3,321],
    'RightHandRing3':      [3,324],
    'RightHandRing_end':      [3,327],

    'RightHandPinky1':       [3,330],
    'RightHandPinky2':      [3,333],
    'RightHandPinky3':      [3,336],
    'RightHandPinky_end':      [3,339],

    'RibbonR1':  [3,342],
    'RibbonR1_end':  [3,345],
    'RibbonR2':  [3,348],
    'RibbonR2_end':  [3,351],
    'RibbonL2':  [3,354],
    'RibbonL2_end':  [3,357],
    
    'LeftUpLeg':       [3,360],
    'LeftLeg':         [3,363],
    'LeftFoot':        [3,366],
    'LeftToe':       [3,369],
    'LeftToe_end':     [3,372],
   
    'RightUpLeg':       [3,375],
    'RightLEg':         [3,378],
    'RightFoot':        [3,381],
    'RightToe':       [3,384],
    'RightToe_end':     [3,387],
    
    'bone_skirtF00': [3, 390],
    'bone_skirtF01': [3, 393],
    'bone_skirtF02': [3, 396],
    'bone_skirtF03': [3, 399],
    'Bone020': [3, 402],
    'Bone026': [3, 405],
    
    'bone_skirtF_R_00': [3, 408],
    'bone_skirtF_R_01': [3, 411],
    'bone_skirtF_R_02': [3, 414],
    'bone_skirtF_R_03': [3, 417],
    'Bone019': [3, 420],
    'Bone028': [3, 423],
    
    'bone_skirtR00': [3, 426],
    'bone_skirtR01': [3, 429],
    'bone_skirtR02': [3, 432],
    'bone_skirtR03': [3, 435],
    'Bone018': [3, 438],
    'Bone029': [3, 441],
    
    'bone_skirtF_L_00': [3, 444],
    'bone_skirtF_L_01': [3, 447],
    'bone_skirtF_L_02': [3, 450],
    'bone_skirtF_L_03': [3, 453],
    'Bone021': [3, 456],
    'Bone027': [3, 459],
    
    'bone_skirtL00': [3, 462],
    'bone_skirtL01': [3, 465],
    'bone_skirtL02': [3, 468],
    'bone_skirtL03': [3, 471],
    'Bone022': [3, 474],
    'Bone033': [3, 477],
    
    'bone_skirtB_L_00': [3, 480],
    'bone_skirtB_L_01': [3, 483],
    'bone_skirtB_L_02': [3, 486],
    'bone_skirtB_L_03': [3, 489],
    'Bone023': [3, 492],
    'Bone032': [3, 495],
    
    'bone_skirtB00': [3, 498],
    'bone_skirtB01': [3, 501],
    'bone_skirtB02': [3, 504],
    'bone_skirtB03': [3, 507],
    'Bone024': [3, 510],
    'Bone031': [3, 513],
    
    'bone_skirtB_R_00': [3, 516],
    'bone_skirtB_R_01': [3, 519],
    'bone_skirtB_R_02': [3, 521],
    'bone_skirtB_R_03': [3, 524],
    'Bone025': [3, 527],
    'Bone030': [3, 530],
        },
        
    "yostar_fullbody_213":{
    'Hips':       3 ,
    'Spine':       3 ,
    'Spine1':        3 ,
    'Chest':       3 ,
    'L_eri':       3 ,
    'R_eri':       3 ,
    'Neck':   3 , 
    'Head':        3 ,
    'Head_end':       3 ,
    
    'LeftEye':  3,
    'LeftEye_end':  3,
    'RightEye':  3,
    'RightEye_end':  3,
    
    'LeftShoulder':    3, 
    'LeftArm':       3, 
    'LeftForearm':     3, 
    'LeftHand':      3, 
    'LeftHandThumb1':     3, 
    'LeftHandThumb2':    3, 
    'LeftHandThumb3':     3, 
    'LeftHandThumb_end':    3, 
    
    'LeftHandIndex1':    3, 
    'LeftHandIndex2':    3, 
    'LeftHandIndex3':  3, 
    'LeftHandIndex_end':    3, 
    
    'LeftHandMiddle1':    3, 
    'LeftHandMiddle2':   3, 
    'LeftHandMiddle3':    3, 
    'LeftHandMiddle_end':    3, 

    'LeftHandRing1':  3, 
    'LeftHandRing2':     3, 
    'LeftHandRing3':     3, 
    'LeftHandRing_end':    3, 

    'LeftHandPinky1':      3, 
    'LeftHandPinky2':     3, 
    'LeftHandPinky3':     3, 
    'LeftHandPinky_end':3, 

    'RightShoulder':   3, 
    'RightArm':        3, 
    'RightForearm':     3, 
    'RightHand':      3, 
    'RightHandThumb1':    3, 
    'RightHandThumb2':     3, 
    'RightHandThumb3':     3, 
    'RightHandThumb_end':     3, 
    
    'RightHandIndex1':      3, 
    'RightHandIndex2':    3, 
    'RightHandIndex3':     3, 
    'RightHandIndex_end':    3, 
    
    'RightHandMiddle1':    3, 
    'RightHandMiddle2':   3, 
    'RightHandMiddle3':      3, 
    'RightHandMiddle_end':    3, 

    'RightHandRing1':     3, 
    'RightHandRing2':    3, 
    'RightHandRing3':      3, 
    'RightHandRing_end':    3, 

    'RightHandPinky1':     3, 
    'RightHandPinky2':   3, 
    'RightHandPinky3':     3, 
    'RightHandPinky_end':    3, 
    
    'LeftUpLeg':       3,
    'LeftLeg':         3,
    'LeftFoot':       3,
    'LeftToe':     3,
    'LeftToe_end':    3,
   
    'RightUpLeg':     3,
    'RightLEg':      3,
    'RightFoot':       3,
    'RightToe':      3,
    'RightToe_end':    3,
        },
    "yostar_mainbody_48": {
    #'Hips':       3 ,
    'Spine':       3 ,
    'Spine1':        3 ,
    'Chest':       3 ,
    'L_eri':       3 ,
    'R_eri':       3 ,
    'Neck':   3 , 
    'Head':        3 ,
    'Head_end':       3 ,
    
    'LeftShoulder':    3, 
    'LeftArm':       3, 
    'LeftForearm':     3, 
    'LeftHand':      3, 

    'RightShoulder':   3, 
    'RightArm':        3, 
    'RightForearm':     3, 
    'RightHand':      3, 
    },
    "yostar_mainbody_69": {
    'Hips':       3 ,
    'Spine':       3 ,
    'Spine1':        3 ,
    'Chest':       3 ,
    'L_eri':       3 ,
    'R_eri':       3 ,
    'Neck':   3 , 
    'Head':        3 ,
    'Head_end':       3 ,
    
    'LeftShoulder':    3, 
    'LeftArm':       3, 
    'LeftForearm':     3, 
    'LeftHand':      3, 

    'RightShoulder':   3, 
    'RightArm':        3, 
    'RightForearm':     3, 
    'RightHand':      3, 
    
    'LeftUpLeg':       3,
    'LeftLeg':         3,
    'LeftFoot':       3,
   
    'RightUpLeg':     3,
    'RightLEg':      3,
    'RightFoot':       3,
    },
    
    "yostar_upbody_168": {
    #'Hips':       3 ,
    'Spine':       3 ,
    'Spine1':        3 ,
    'Chest':       3 ,
    'L_eri':       3 ,
    'R_eri':       3 ,
    'Neck':   3 , 
    'Head':        3 ,
    'Head_end':       3 ,
    
    'LeftShoulder':    3, 
    'LeftArm':       3, 
    'LeftForearm':     3, 
    'LeftHand':      3, 
    'LeftHandThumb1':     3, 
    'LeftHandThumb2':    3, 
    'LeftHandThumb3':     3, 
    'LeftHandThumb_end':    3, 
    
    'LeftHandIndex1':    3, 
    'LeftHandIndex2':    3, 
    'LeftHandIndex3':  3, 
    'LeftHandIndex_end':    3, 
    
    'LeftHandMiddle1':    3, 
    'LeftHandMiddle2':   3, 
    'LeftHandMiddle3':    3, 
    'LeftHandMiddle_end':    3, 

    'LeftHandRing1':  3, 
    'LeftHandRing2':     3, 
    'LeftHandRing3':     3, 
    'LeftHandRing_end':    3, 

    'LeftHandPinky1':      3, 
    'LeftHandPinky2':     3, 
    'LeftHandPinky3':     3, 
    'LeftHandPinky_end':3, 
        
    'RightShoulder':   3, 
    'RightArm':        3, 
    'RightForearm':     3, 
    'RightHand':      3, 
    'RightHandThumb1':    3, 
    'RightHandThumb2':     3, 
    'RightHandThumb3':     3, 
    'RightHandThumb_end':     3, 
    
    'RightHandIndex1':      3, 
    'RightHandIndex2':    3, 
    'RightHandIndex3':     3, 
    'RightHandIndex_end':    3, 
    
    'RightHandMiddle1':    3, 
    'RightHandMiddle2':   3, 
    'RightHandMiddle3':      3, 
    'RightHandMiddle_end':    3, 

    'RightHandRing1':     3, 
    'RightHandRing2':    3, 
    'RightHandRing3':      3, 
    'RightHandRing_end':    3, 

    'RightHandPinky1':     3, 
    'RightHandPinky2':   3, 
    'RightHandPinky3':     3, 
    'RightHandPinky_end':    3, 
    },
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
    '''
    todo
    '''
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
        #print(mu1[0], mu2[0])
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        #print(sigma1[0], sigma2[0])
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        #print(diff, covmean[0])
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
    elif "yostar" in pose_version:
        ori_list = joints_list["yostar"]
        target_list = joints_list[pose_version]
        file_content_length = 1056
    else:
        ori_list = joints_list["japanese_joints"]
        target_list = joints_list[pose_version]
        file_content_length = 366
    
    bvh_files_dirs = sorted(glob.glob(f'{res_bvhlist}*.bvh'), key=str)
    #test_seq_list = os.list_dir(demo_name).sort()
    
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