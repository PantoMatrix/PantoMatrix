import librosa
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.pyplot import figure
import math
from scipy.signal import argrelextrema


class L1div(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0
    def run(self, results):
        self.counter += results.shape[0]
        mean = np.mean(results, 0)
        for i in range(results.shape[0]):
            results[i, :] = abs(results[i, :] - mean)
        sum_l1 = np.sum(results)
        self.sum += sum_l1
    def avg(self):
        return self.sum/self.counter
        

class SRGR(object):
    def __init__(self, threshold=0.1, joints=47):
        self.threshold = threshold
        self.pose_dimes = joints
        self.counter = 0
        self.sum = 0
        
    def run(self, results, targets, semantic):
        results = results.reshape(-1, self.pose_dimes, 3)
        targets = targets.reshape(-1, self.pose_dimes, 3)
        semantic = semantic.reshape(-1)
        diff = np.sum(abs(results-targets),2)
        success = np.where(diff<self.threshold, 1.0, 0.0)
        for i in range(success.shape[0]):
            # srgr == 0.165 when all success, scale range to [0, 1]
            success[i, :] *= semantic[i] * (1/0.165) 
        rate = np.sum(success)/(success.shape[0]*success.shape[1])
        self.counter += success.shape[0]
        self.sum += (rate*success.shape[0])
        return rate
    
    def avg(self):
        return self.sum/self.counter

class alignment(object):
    def __init__(self, sigma, order):
        self.sigma = sigma
        self.order = order
        self.times = self.oenv = self.S = self.rms = None
        self.pose_data = []
    
    def load_audio(self, wave, t_start, t_end, without_file=False, sr_audio=16000):
        if without_file:
            y = wave
            sr = sr_audio
        else: y, sr = librosa.load(wave)
        short_y = y[t_start*sr:t_end*sr]
        self.oenv = librosa.onset.onset_strength(y=short_y, sr=sr)
        self.times = librosa.times_like(self.oenv)
        # Detect events without backtracking
        onset_raw = librosa.onset.onset_detect(onset_envelope=self.oenv, backtrack=False)
        onset_bt = librosa.onset.onset_backtrack(onset_raw, self.oenv)
        self.S = np.abs(librosa.stft(y=short_y))
        self.rms = librosa.feature.rms(S=self.S)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, self.rms[0])
        return onset_raw, onset_bt, onset_bt_rms

    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        data_each_file = []
        if without_file:
            for line_data_np in pose: #,args.pre_frames, args.pose_length
                data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121], ],0))
        else: 
            with open(pose, "r") as f:
                for i, line_data in enumerate(f.readlines()):
                    if i < 432: continue
                    line_data_np = np.fromstring(line_data, sep=" ",)
                    if pose_fps == 15:
                        if i % 2 == 0:
                            continue
                    data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121], ],0))
        data_each_file = np.array(data_each_file)
        vel= data_each_file[1:, :] - data_each_file[:-1, :]
        # l1 
        # vel_rigth_shoulder = abs(vel[:, 0]) + abs(vel[:, 1]) + abs(vel[:, 2])
        # vel_rigth_arm = abs(vel[:, 3]) + abs(vel[:, 4]) + abs(vel[:, 5])
        # vel_rigth_wrist = abs(vel[:, 6]) + abs(vel[:, 7]) + abs(vel[:, 8])
        # l2
        vel_right_shoulder = np.linalg.norm(np.array([vel[:, 0], vel[:, 1], vel[:, 2]]), axis=0)
        vel_right_arm = np.linalg.norm(np.array([vel[:, 3], vel[:, 4], vel[:, 5]]), axis=0)
        vel_right_wrist = np.linalg.norm(np.array([vel[:, 6], vel[:, 7], vel[:, 8]]), axis=0)
        beat_right_arm = argrelextrema(vel_right_arm[t_start*pose_fps:t_end*pose_fps], np.less, order=self.order)
        beat_right_shoulder = argrelextrema(vel_right_shoulder[t_start*pose_fps:t_end*pose_fps], np.less, order=self.order)
        beat_right_wrist = argrelextrema(vel_right_wrist[t_start*pose_fps:t_end*pose_fps], np.less, order=self.order)
        vel_left_shoulder = np.linalg.norm(np.array([vel[:, 9], vel[:, 10], vel[:, 11]]), axis=0)
        vel_left_arm = np.linalg.norm(np.array([vel[:, 12], vel[:, 13], vel[:, 14]]), axis=0)
        vel_left_wrist = np.linalg.norm(np.array([vel[:, 15], vel[:, 16], vel[:, 17]]), axis=0)
        beat_left_arm = argrelextrema(vel_left_arm, np.less, order=self.order)
        beat_left_shoulder = argrelextrema(vel_left_shoulder, np.less, order=self.order)
        beat_left_wrist = argrelextrema(vel_left_wrist, np.less, order=self.order)
        return beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist
    
    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, t_start, t_end, pose_fps)
        return onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist 

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, i, i+dur, pose_fps)
            dis_all_b2a= self.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist)
            print(f"{i}s: ",dis_all_b2a)

    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(self.S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0])
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label='Onset strength')
        ax[1].vlines(librosa.frames_to_time(onset_raw), 0, self.oenv.max(), label='Raw onsets', color='r')
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label='Onset strength')
        ax[2].vlines(librosa.frames_to_time(onset_bt), 0, self.oenv.max(), label='Backtracked', color='r')
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label='RMS')
        ax[3].vlines(librosa.frames_to_time(onset_bt_rms), 0, self.oenv.max(), label='Backtracked (RMS)', color='r')
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)
    
    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel[0]/pose_fps + offset 
        return time_vel    
    
    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
        dis_all_b2a = 0
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2)/(2*sigma**2))
        dis_all_b2a /= len(b)
        return dis_all_b2a 

    def calculate_align(self, onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, pose_fps=15):
        # more stable solution
        # avg_dis_all_b2a = 0
        # for audio_beat in [onset_raw, onset_bt, onset_bt_rms]:
        #     for pose_beat in [beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist]:
        #         audio_bt = librosa.frames_to_time(audio_beat)
        #         pose_bt = self.motion_frames2time(pose_beat, 0, pose_fps)
        #         dis_all_b2a = self.GAHR(pose_bt, audio_bt, self.sigma)
        #         avg_dis_all_b2a += dis_all_b2a
        # avg_dis_all_b2a /= 18
        audio_bt = librosa.frames_to_time(onset_bt_rms)
        pose_bt = self.motion_frames2time(beat_right_wrist, 0, pose_fps)
        avg_dis_all_b2a = self.GAHR(pose_bt, audio_bt, self.sigma)
        return avg_dis_all_b2a  