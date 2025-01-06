import os
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

json_path = "./datasets/data_json/beat2_s20_l128_speaker2.json"
with open(json_path, 'r') as f:
    data = json.load(f)

arr = []
for d in tqdm(data):
    m = np.load(d["motion_path"].replace("/content/beat_v2.0.0/", "./BEAT2/"))["poses"][d["start_idx"]:d["end_idx"]]
    arr.append(m)
arr = np.array(arr).reshape(len(arr), 128, 55, 3)[:, :, :21]

X_content = arr.reshape(len(arr), -1)
content_km = KMeans(n_clusters=10, random_state=0).fit(X_content)
content_labels = content_km.labels_
for i, d in tqdm(enumerate(data)):
    d["content_label"] = int(content_labels[i])

unique_c, counts_c = np.unique(content_labels, return_counts=True)
for uc, cc in zip(unique_c, counts_c):
    print(uc, cc, round(cc/len(content_labels), 2))

vel = np.diff(arr, axis=1)
mag = np.linalg.norm(vel, axis=-1)
beat = np.zeros_like(mag)
w = 5
for i in tqdm(range(beat.shape[0])):
    for j in range(beat.shape[2]):
        for t in range(w, beat.shape[1]-w):
            if mag[i, t, j] == np.min(mag[i, t-w:t+w+1, j]):
                beat[i, t, j] = 1
X_rhythm = beat.reshape(len(beat), -1)
rhythm_km = KMeans(n_clusters=10, random_state=0).fit(X_rhythm)
rhythm_labels = rhythm_km.labels_
for i, d in enumerate(data):
    d["rhythm_label"] = int(rhythm_labels[i])

unique_r, counts_r = np.unique(rhythm_labels, return_counts=True)
for ur, cr in zip(unique_r, counts_r):
    print(ur, cr, round(cr/len(rhythm_labels), 2))

with open("./datasets/data_json/beat2_s20_l128_speaker2_disco.json", 'w') as f:
    json.dump(data, f)
