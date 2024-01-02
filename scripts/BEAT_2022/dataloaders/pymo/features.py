'''
A set of mocap feature extraction functions

Created by Omid Alemi | Nov 17 2017

'''
import numpy as np
import pandas as pd
import peakutils
import matplotlib.pyplot as plt

def get_foot_contact_idxs(signal, t=0.02, min_dist=120):
    up_idxs = peakutils.indexes(signal, thres=t/max(signal), min_dist=min_dist)
    down_idxs = peakutils.indexes(-signal, thres=t/min(signal), min_dist=min_dist)

    return [up_idxs, down_idxs]


def create_foot_contact_signal(mocap_track, col_name, start=1, t=0.02, min_dist=120):
    signal = mocap_track.values[col_name].values
    idxs = get_foot_contact_idxs(signal, t, min_dist)

    step_signal = []

    c = start
    for f in range(len(signal)):    
        if f in idxs[1]:
            c = 0
        elif f in idxs[0]:
            c = 1
                                            
        step_signal.append(c)
    
    return step_signal

def plot_foot_up_down(mocap_track, col_name, t=0.02, min_dist=120):
    
    signal = mocap_track.values[col_name].values
    idxs = get_foot_contact_idxs(signal, t, min_dist)

    plt.plot(mocap_track.values.index, signal)
    plt.plot(mocap_track.values.index[idxs[0]], signal[idxs[0]], 'ro')
    plt.plot(mocap_track.values.index[idxs[1]], signal[idxs[1]], 'go')
