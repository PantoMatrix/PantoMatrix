'''
BVH Parser Class

By Omid Alemi
Created: June 12, 2017

Based on: https://gist.github.com/johnfredcee/2007503

'''
import re
import numpy as np
from .data import Joint, MocapData

class BVHScanner():
    '''
    A wrapper class for re.Scanner
    '''
    def __init__(self):

        def identifier(scanner, token):
            return 'IDENT', token

        def operator(scanner, token):
            return 'OPERATOR', token

        def digit(scanner, token):
            return 'DIGIT', token

        def open_brace(scanner, token):
            return 'OPEN_BRACE', token

        def close_brace(scanner, token):
            return 'CLOSE_BRACE', token

        self.scanner = re.Scanner([
            (r'[a-zA-Z_]\w*', identifier),
            #(r'-*[0-9]+(\.[0-9]+)?', digit), # won't work for .34
            #(r'[-+]?[0-9]*\.?[0-9]+', digit), # won't work for 4.56e-2
            #(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', digit),
            (r'-*[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', digit),
            (r'}', close_brace),
            (r'}', close_brace),
            (r'{', open_brace),
            (r':', None),
            (r'\s+', None)
        ])

    def scan(self, stuff):
        return self.scanner.scan(stuff)



class BVHParser():
    '''
    A class to parse a BVH file.
    
    Extracts the skeleton and channel values
    '''
    def __init__(self, filename=None):
        self.reset()

    def reset(self): 
        self._skeleton = {}
        self.bone_context = []
        self._motion_channels = []
        self._motions = []
        self.current_token = 0
        self.framerate = 0.0
        self.root_name = ''

        self.scanner = BVHScanner()
        
        self.data = MocapData()


    def parse(self, filename, start=0, stop=-1):
        self.reset()

        with open(filename, 'r') as bvh_file:
            raw_contents = bvh_file.read()
        tokens, remainder = self.scanner.scan(raw_contents)
        self._parse_hierarchy(tokens)
        self.current_token = self.current_token + 1
        self._parse_motion(tokens, start, stop)
        
        self.data.skeleton = self._skeleton
        self.data.channel_names = self._motion_channels
        self.data.values = self._to_DataFrame()
        self.data.root_name = self.root_name
        self.data.framerate = self.framerate

        return self.data
    
    def _to_DataFrame(self):
        '''Returns all of the channels parsed from the file as a pandas DataFrame'''

        import pandas as pd
        time_index = pd.to_timedelta([f[0] for f in self._motions], unit='s')
        frames = [f[1] for f in self._motions]
        channels = np.asarray([[channel[2] for channel in frame] for frame in frames])
        column_names = ['%s_%s'%(c[0], c[1]) for c in self._motion_channels]

        return pd.DataFrame(data=channels, index=time_index, columns=column_names)


    def _new_bone(self, parent, name):
        bone = {'parent': parent, 'channels': [], 'offsets': [], 'order': '','children': []}
        return bone

    def _push_bone_context(self,name):
        self.bone_context.append(name)

    def _get_bone_context(self):
        return self.bone_context[len(self.bone_context)-1]

    def _pop_bone_context(self):
        self.bone_context = self.bone_context[:-1]
        return self.bone_context[len(self.bone_context)-1]

    def _read_offset(self, bvh, token_index):
        if bvh[token_index] != ('IDENT', 'OFFSET'):
            return None, None
        token_index = token_index + 1
        offsets = [0.0] * 3
        for i in range(3):
            offsets[i] = float(bvh[token_index][1])
            token_index = token_index + 1
        return offsets, token_index
    
    def _read_channels(self, bvh, token_index):
        if bvh[token_index] != ('IDENT', 'CHANNELS'):
            return None, None
        token_index = token_index + 1
        channel_count = int(bvh[token_index][1])
        token_index = token_index + 1
        channels = [""] * channel_count
        order = ""
        for i in range(channel_count):
            channels[i] = bvh[token_index][1]
            token_index = token_index + 1
            if(channels[i] == "Xrotation" or channels[i]== "Yrotation" or channels[i]== "Zrotation"):
                order += channels[i][0]
            else :
                order = ""
        return channels, token_index, order

    def _parse_joint(self, bvh, token_index):
        end_site = False
        joint_id = bvh[token_index][1]
        token_index = token_index + 1
        joint_name = bvh[token_index][1]
        token_index = token_index + 1
        
        parent_name = self._get_bone_context()

        if (joint_id == "End"):
            joint_name = parent_name+ '_Nub'
            end_site = True
        joint = self._new_bone(parent_name, joint_name)
        if bvh[token_index][0] != 'OPEN_BRACE':
            print('Was expecting brance, got ', bvh[token_index])
            return None
        token_index = token_index + 1
        offsets, token_index = self._read_offset(bvh, token_index)
        joint['offsets'] = offsets
        if not end_site:
            channels, token_index, order = self._read_channels(bvh, token_index)
            joint['channels'] = channels
            joint['order'] = order
            for channel in channels:
                self._motion_channels.append((joint_name, channel))

        self._skeleton[joint_name] = joint
        self._skeleton[parent_name]['children'].append(joint_name)

        while (bvh[token_index][0] == 'IDENT' and bvh[token_index][1] == 'JOINT') or  (bvh[token_index][0] == 'IDENT' and bvh[token_index][1] == 'End'):
            self._push_bone_context(joint_name)
            token_index = self._parse_joint(bvh, token_index)
            self._pop_bone_context()

        if bvh[token_index][0] == 'CLOSE_BRACE':
            return token_index + 1

        print('Unexpected token ', bvh[token_index])

    def _parse_hierarchy(self, bvh):
        self.current_token = 0
        if bvh[self.current_token] != ('IDENT', 'HIERARCHY'):
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token] != ('IDENT', 'ROOT'):
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][0] != 'IDENT':
            return None

        root_name = bvh[self.current_token][1]
        root_bone = self._new_bone(None, root_name)
        self.current_token = self.current_token + 2 #skipping open brace
        offsets, self.current_token = self._read_offset(bvh, self.current_token)
        channels, self.current_token, order = self._read_channels(bvh, self.current_token)
        root_bone['offsets'] = offsets
        root_bone['channels'] = channels
        root_bone['order'] = order
        self._skeleton[root_name] = root_bone
        self._push_bone_context(root_name)
        
        for channel in channels:
            self._motion_channels.append((root_name, channel))

        while bvh[self.current_token][1] == 'JOINT':
            self.current_token = self._parse_joint(bvh, self.current_token)
        
        self.root_name = root_name

    def _parse_motion(self, bvh, start, stop):
        if bvh[self.current_token][0] != 'IDENT':
            print('Unexpected text')
            return None
        if bvh[self.current_token][1] != 'MOTION':
            print('No motion section')
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][1] != 'Frames':
            return None
        self.current_token = self.current_token + 1
        frame_count = int(bvh[self.current_token][1])
        
        if stop<0 or stop>frame_count:
            stop = frame_count
            
        assert(start>=0)
        assert(start<stop)
        
        self.current_token = self.current_token + 1
        if bvh[self.current_token][1] != 'Frame':
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][1] != 'Time':
            return None
        self.current_token = self.current_token + 1
        frame_rate = float(bvh[self.current_token][1])

        self.framerate = frame_rate
       
        self.current_token = self.current_token + 1
       
        frame_time = 0.0
        self._motions = [()] * (stop-start)
        idx=0
        for i in range(stop):
            channel_values = []
            for channel in self._motion_channels:
                channel_values.append((channel[0], channel[1], float(bvh[self.current_token][1])))
                self.current_token = self.current_token + 1

            if i>=start:
                self._motions[idx] = (frame_time, channel_values)
                frame_time = frame_time + frame_rate
                idx+=1
