import os
import numpy as np
import torch
import smplx
from tqdm import tqdm

root_dir = './BEAT2/beat_english_v2.0.0/smplxflame_30'
output_dir = "./BEAT2/beat_english_v2.0.0/footcontact"
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smplx_model = smplx.create(
    "./emage_evaltools/smplx_models/",
    model_type='smplx',
    gender='NEUTRAL_2020',
    use_face_contour=False,
    num_betas=300,
    num_expression_coeffs=100,
    ext='npz',
    use_pca=False,
).eval().to(device)

max_length = 128
for data_file in tqdm(os.listdir(root_dir)):
    if not data_file.endswith(".npz"):
        continue
    data = np.load(os.path.join(root_dir, data_file), allow_pickle=True)
    betas = data["betas"]
    poses = data["poses"]
    trans = data["trans"]
    if "expressions" in data:
        exps = data["expressions"]
    else:
        exps = np.zeros((poses.shape[0], 100))
    n, c = poses.shape
    betas = betas.reshape(1, 300)
    betas = np.tile(betas, (n, 1))
    betas = torch.from_numpy(betas).float().to(device)
    poses = torch.from_numpy(poses.reshape(n, c)).float().to(device)
    exps = torch.from_numpy(exps.reshape(n, 100)).float().to(device)
    trans = torch.from_numpy(trans.reshape(n, 3)).float().to(device)
    s, r = n//max_length, n%max_length
    all_tensor = []
    for i in range(s):
        with torch.no_grad():
            joints = model(
                betas=betas[i*max_length:(i+1)*max_length],
                transl=trans[i*max_length:(i+1)*max_length],
                expression=exps[i*max_length:(i+1)*max_length],
                jaw_pose=poses[i*max_length:(i+1)*max_length,66:69],
                global_orient=poses[i*max_length:(i+1)*max_length,:3],
                body_pose=poses[i*max_length:(i+1)*max_length,3:66],
                left_hand_pose=poses[i*max_length:(i+1)*max_length,75:120],
                right_hand_pose=poses[i*max_length:(i+1)*max_length,120:165],
                leye_pose=poses[i*max_length:(i+1)*max_length,69:72],
                reye_pose=poses[i*max_length:(i+1)*max_length,72:75],
                return_joints=True
            )['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
        all_tensor.append(joints)
    if r != 0:
        with torch.no_grad():
            joints = model(
                betas=betas[s*max_length:s*max_length+r],
                transl=trans[s*max_length:s*max_length+r],
                expression=exps[s*max_length:s*max_length+r],
                jaw_pose=poses[s*max_length:s*max_length+r,66:69],
                global_orient=poses[s*max_length:s*max_length+r,:3],
                body_pose=poses[s*max_length:s*max_length+r,3:66],
                left_hand_pose=poses[s*max_length:s*max_length+r,75:120],
                right_hand_pose=poses[s*max_length:s*max_length+r,120:165],
                leye_pose=poses[s*max_length:s*max_length+r,69:72],
                reye_pose=poses[s*max_length:s*max_length+r,72:75],
                return_joints=True
            )['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
        all_tensor.append(joints)
    joints = torch.cat(all_tensor, axis=0)
    feetv = torch.zeros(joints.shape[1], joints.shape[0])
    joints = joints.permute(1, 0, 2)
    feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
    contacts = (feetv < 0.01).numpy().astype(float)
    contacts = contacts.transpose(1, 0)
    np.save(os.path.join(output_dir, data_file.replace(".npz", ".npy")), contacts)
