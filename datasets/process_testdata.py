import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

# Example parameters
stride = 20
motion_length = 64
speaker_target = 2
use_additional = False

root_dir = './beat_english_v2.0.0/'
output_dir = "./datasets/data_json/"
os.makedirs(output_dir, exist_ok=True)
train_test_split_path = './beat_english_v2.0.0/train_test_split.csv'
df = pd.read_csv(train_test_split_path)

filtered_df = df[(df['id'].str.split('_').str[0].astype(int) == speaker_target) & (df['type'] != 'additional')]
clips = []
for idx, row_item in tqdm(filtered_df.iterrows()):
    video_id = row_item['id']
    mode = row_item['type'] 
    # check exist
    npz_path = os.path.join(root_dir, "smplxflame_30", video_id + ".npz")
    wav_path = os.path.join(root_dir, "wave16k", video_id + ".wav")

    try:
      motion_data = np.load(npz_path, allow_pickle=True)
    except:
      print(f"cant open {npz_path}")
    
    try:
      wave_data, _ = librosa.load(wav_path, sr=None)
    except:
      print(f"cant open {wav_path}")

    motion = motion_data['poses']
    total_len = motion.shape[0]

    for i in range(0, total_len - motion_length, stride):
        clip = {
            "video_id": video_id,
            "motion_path": npz_path,
            "audio_path": wav_path,
            "mode": mode,
            "start_idx": i,
            "end_idx": i + motion_length
        }
        clips.append(clip)

output_json = os.path.join(output_dir, f"beat2_s{stride}_l{motion_length}_speaker{speaker_target}.json")
with open(output_json, 'w') as f:
    json.dump(clips, f, indent=4)
