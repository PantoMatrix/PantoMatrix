"""
Thanks to the author of this API 
Tomoya Akiyama: https://research.cyberagent.ai/people/tomoya_akiyama/
"""

import math
import cv2
import numpy as np
import torch
import smplx
from pytorch3d.renderer import PerspectiveCameras
from torchvision.io import write_video
from torchvision.transforms.functional import convert_image_dtype

SMPLX_BODY_JOINT_EDGES = [
    {"indices": [12, 17], "color": [255, 0, 0]},
    {"indices": [12, 16], "color": [255, 85, 0]},
    {"indices": [17, 19], "color": [255, 170, 0]},
    {"indices": [19, 21], "color": [255, 255, 0]},
    {"indices": [16, 18], "color": [170, 255, 0]},
    {"indices": [18, 20], "color": [85, 255, 0]},
    {"indices": [2, 12], "color": [0, 255, 0]},
    {"indices": [2, 5], "color": [0, 255, 85]},
    {"indices": [5, 8], "color": [0, 255, 170]},
    {"indices": [1, 12], "color": [0, 255, 255]},
    {"indices": [1, 4], "color": [0, 170, 255]},
    {"indices": [4, 7], "color": [0, 85, 255]},
    {"indices": [12, 55], "color": [0, 0, 255]},
    {"indices": [55, 56], "color": [85, 0, 255]},
    {"indices": [56, 58], "color": [170, 0, 255]},
    {"indices": [55, 57], "color": [255, 0, 255]},
    {"indices": [57, 59], "color": [255, 0, 170]},
]
SMPLX_BODY_JOINTS = [
    {"index": 55, "color": [255, 0, 0]},
    {"index": 12, "color": [255, 85, 0]},
    {"index": 17, "color": [255, 170, 0]},
    {"index": 19, "color": [255, 255, 0]},
    {"index": 21, "color": [170, 255, 0]},
    {"index": 16, "color": [85, 255, 0]},
    {"index": 18, "color": [0, 255, 0]},
    {"index": 20, "color": [0, 255, 85]},
    {"index": 2, "color": [0, 255, 170]},
    {"index": 5, "color": [0, 255, 255]},
    {"index": 8, "color": [0, 170, 255]},
    {"index": 1, "color": [0, 85, 255]},
    {"index": 4, "color": [0, 0, 255]},
    {"index": 7, "color": [85, 0, 255]},
    {"index": 56, "color": [170, 0, 255]},
    {"index": 57, "color": [255, 0, 255]},
    {"index": 58, "color": [255, 0, 170]},
    {"index": 59, "color": [255, 0, 85]},
]
SMPLX_HAND_JOINT_EDGES = [
    {"indices": [21, 52], "color": [255, 0, 0]},
    {"indices": [52, 53], "color": [255, 76, 0]},
    {"indices": [53, 54], "color": [255, 153, 0]},
    {"indices": [54, 71], "color": [255, 229, 0]},
    {"indices": [21, 40], "color": [204, 255, 0]},
    {"indices": [40, 41], "color": [128, 255, 0]},
    {"indices": [41, 42], "color": [51, 255, 0]},
    {"indices": [42, 72], "color": [0, 255, 26]},
    {"indices": [21, 43], "color": [0, 255, 102]},
    {"indices": [43, 44], "color": [0, 255, 179]},
    {"indices": [44, 45], "color": [0, 255, 255]},
    {"indices": [45, 73], "color": [0, 179, 255]},
    {"indices": [21, 49], "color": [0, 102, 255]},
    {"indices": [49, 50], "color": [0, 26, 255]},
    {"indices": [50, 51], "color": [51, 0, 255]},
    {"indices": [51, 74], "color": [128, 0, 255]},
    {"indices": [21, 46], "color": [204, 0, 255]},
    {"indices": [46, 47], "color": [255, 0, 230]},
    {"indices": [47, 48], "color": [255, 0, 153]},
    {"indices": [48, 75], "color": [255, 0, 77]},
    {"indices": [20, 37], "color": [255, 0, 0]},
    {"indices": [37, 38], "color": [255, 76, 0]},
    {"indices": [38, 39], "color": [255, 153, 0]},
    {"indices": [39, 66], "color": [255, 229, 0]},
    {"indices": [20, 25], "color": [204, 255, 0]},
    {"indices": [25, 26], "color": [128, 255, 0]},
    {"indices": [26, 27], "color": [51, 255, 0]},
    {"indices": [27, 67], "color": [0, 255, 26]},
    {"indices": [20, 28], "color": [0, 255, 102]},
    {"indices": [28, 29], "color": [0, 255, 179]},
    {"indices": [29, 30], "color": [0, 255, 255]},
    {"indices": [30, 68], "color": [0, 179, 255]},
    {"indices": [20, 34], "color": [0, 102, 255]},
    {"indices": [34, 35], "color": [0, 26, 255]},
    {"indices": [35, 36], "color": [51, 0, 255]},
    {"indices": [36, 69], "color": [128, 0, 255]},
    {"indices": [20, 31], "color": [204, 0, 255]},
    {"indices": [31, 32], "color": [255, 0, 230]},
    {"indices": [32, 33], "color": [255, 0, 153]},
    {"indices": [33, 70], "color": [255, 0, 77]},
]
SMPLX_HAND_JOINTS = [20, 21] + list(range(25, 55)) + list(range(66, 76))
SMPLX_FACE_LANDMARKS = list(range(76, 144))

def _draw_bodypose(canvas, joints_np):
    c = canvas.copy()
    for edge_dict in SMPLX_BODY_JOINT_EDGES:
        i = edge_dict["indices"]
        color = edge_dict["color"]
        xy = joints_np[i]
        center = np.mean(xy, axis=0).astype(int)
        length = np.linalg.norm(xy[0] - xy[1])
        angle = math.degrees(math.atan2(xy[0, 1] - xy[1, 1], xy[0, 0] - xy[1, 0]))
        polygon = cv2.ellipse2Poly(center, (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(c, polygon, color)
    c = (c * 0.6).astype(np.uint8)
    for j_info in SMPLX_BODY_JOINTS:
        center = joints_np[j_info["index"]].astype(int)
        cv2.circle(c, tuple(center), 4, (255, 255, 255), -1)
    return c

def _draw_handpose(canvas, joints_np):
    c = canvas.copy()
    for edge_dict in SMPLX_HAND_JOINT_EDGES:
        i = edge_dict["indices"]
        color = edge_dict["color"]
        xy = joints_np[i].astype(int)
        if xy.min() > 0:
            cv2.line(c, tuple(xy[0]), tuple(xy[1]), color, 2)
    for j_idx in SMPLX_HAND_JOINTS:
        center = joints_np[j_idx].astype(int)
        if center.min() > 0:
            cv2.circle(c, tuple(center), 4, (0, 0, 255), -1)
    return c

def _draw_facepose(canvas, joints_np):
    c = canvas.copy()
    for j_idx in SMPLX_FACE_LANDMARKS:
        center = joints_np[j_idx].astype(int)
        if center.min() > 0:
            cv2.circle(c, tuple(center), 3, (255, 255, 255), -1)
    return c

def _draw_joints_2d(joints_2d, height, width, face_only):
    outputs = []
    for j2d in joints_2d:
        # Convert each frame's joints to NumPy
        j2d_np = j2d.detach().cpu().numpy()
        c = np.zeros((height, width, 3), dtype=np.uint8)
        if face_only:
            c = _draw_facepose(c, j2d_np)
        else:
            c = _draw_bodypose(c, j2d_np)
            c = _draw_handpose(c, j2d_np)
            c = _draw_facepose(c, j2d_np)
        outputs.append(convert_image_dtype(torch.tensor(c, dtype=torch.uint8), torch.uint8))
    return torch.stack(outputs).permute(0, 3, 1, 2)

def _draw_joints_3d(joints_3d, height, width, face_only):
    outputs = []
    for j3d in joints_3d:
        xy = j3d[:, :2].detach().cpu().numpy().copy()
        z = j3d[:, 2].detach().cpu().numpy().copy()
        z_min, z_max = z.min(), z.max()
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)

        # Normalize XY to fit in the image
        xy[:, 0] = (xy[:, 0] - xy[:, 0].min()) / (xy[:, 0].max() - xy[:, 0].min() + 1e-8) * (width - 1)
        xy[:, 1] = (xy[:, 1] - xy[:, 1].min()) / (xy[:, 1].max() - xy[:, 1].min() + 1e-8) * (height - 1)

        c = np.zeros((height, width, 3), dtype=np.uint8)
        # j2d: [num_joints, 3], last dim is the normalized z
        j2d = np.hstack([xy, z_norm.reshape(-1, 1)])
        if face_only:
            c = _draw_facepose(c, j2d)
        else:
            c = _draw_bodypose(c, j2d)
            c = _draw_handpose(c, j2d)
            c = _draw_facepose(c, j2d)

        outputs.append(convert_image_dtype(torch.tensor(c, dtype=torch.uint8), torch.uint8))
    return torch.stack(outputs).permute(0, 3, 1, 2)

def _load_motion_dict(
    motion_dict,
    device,
    remove_global=False,
    face_only=False
):
    n = motion_dict["poses"].shape[0]
    smplx_inputs = {
        "betas": torch.tensor(motion_dict["betas"]).view(1, -1),
        "global_orient": torch.tensor(motion_dict["poses"][:, :3]),
        "body_pose": torch.tensor(motion_dict["poses"][:, 3 : 22 * 3]),
        "left_hand_pose": torch.tensor(motion_dict["poses"][:, 25 * 3 : 40 * 3]),
        "right_hand_pose": torch.tensor(motion_dict["poses"][:, 40 * 3 : 55 * 3]),
        "transl": torch.tensor(motion_dict["trans"]),
        "expression": torch.tensor(motion_dict["expressions"]),
        "jaw_pose": torch.tensor(motion_dict["poses"][:, 22 * 3 : 23 * 3]),
        "leye_pose": torch.tensor(motion_dict["poses"][:, 23 * 3 : 24 * 3]),
        "reye_pose": torch.tensor(motion_dict["poses"][:, 24 * 3 : 25 * 3]),
    }
    # Move everything to device
    for k, v in smplx_inputs.items():
        smplx_inputs[k] = v.to(device=device, dtype=torch.float32)

    # 1) If remove_global == True, keep 'transl' at the first frame's value for all frames
    if remove_global:
        first_frame_trans = smplx_inputs["transl"][0].clone()
        smplx_inputs["transl"][:] = first_frame_trans

    # 2) If face_only == True, zero out everything but the jaw pose
    if face_only:
        smplx_inputs["global_orient"][:] = 0.0
        smplx_inputs["body_pose"][:] = 0.0
        smplx_inputs["left_hand_pose"][:] = 0.0
        smplx_inputs["right_hand_pose"][:] = 0.0
        smplx_inputs["leye_pose"][:] = 0.0
        smplx_inputs["reye_pose"][:] = 0.0
        # The jaw_pose and expression remain as is (allowing mouth movements),
        # so the head is "frozen" in place except for jaw animation.

    return n, smplx_inputs

def _get_smplx_model(smplx_folder, batch_size, device):
    smplx_model = smplx.create(
        model_path=smplx_folder,
        model_type="smplx",
        gender="NEUTRAL_2020",
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=False,
        use_hands=True,
        use_face=True,
        num_betas=300,
        num_expression_coeffs=100,
        batch_size=batch_size,
        dtype=torch.float32,
    ).to(device)
    return smplx_model.eval()

def _get_cameras(
    batch_size,
    height,
    width,
    focal_length,
    camera_transl,
    device
):
    r = torch.tensor(
        [[-1, 0, 0],
         [ 0, 1, 0],
         [ 0, 0, 1]], 
        device=device, dtype=torch.float32
    )
    t = torch.tensor(camera_transl, device=device, dtype=torch.float32)
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=((width / 2, height / 2),),
        in_ndc=False,
        R=r.expand(batch_size, -1, -1),
        T=t.expand(batch_size, -1),
        image_size=((height, width),),
        device=device,
    )
    return cameras

# New fix code snippet (inside render2d or render3d):
def render2d(
    motion_dict,
    resolution=(512, 512),
    face_only=False,
    remove_global=False,
    smplx_folder="./emage_evaltools/smplx_models/",
    focal_length=5000.0,
    camera_transl=(0.0, -0.8, 16.0),
    device=torch.device("cuda"),
):
    h, w = resolution
    # for face-only, override camera to zoom in
    if face_only:
        camera_transl = (0.0, -1.55, 6.0)
        focal_length = 10000.0
    n, smplx_inputs = _load_motion_dict(
        motion_dict, device, remove_global=remove_global, face_only=face_only
    )
    model = _get_smplx_model(smplx_folder, n, device)
    out = model(**smplx_inputs)
    cams = _get_cameras(n, h, w, focal_length, camera_transl, device)
    j2d = cams.transform_points_screen(out.joints)[:, :, :2]
    frames_2d = _draw_joints_2d(j2d, h, w, face_only)
    return frames_2d

def render3d(
    motion_dict,
    resolution=(512, 512),
    face_only=False,
    remove_global=False,
    smplx_folder="./emage_evaltools/smplx_models/",
    device=torch.device("cuda"),
):
    h, w = resolution
    n, smplx_inputs = _load_motion_dict(
        motion_dict,
        device,
        remove_global=remove_global,
        face_only=face_only
    )
    model = _get_smplx_model(smplx_folder, n, device)
    out = model(**smplx_inputs)
    frames_3d = _draw_joints_3d(out.joints, h, w, face_only)
    return frames_3d

def example_usage():
    # Suppose we have an NPZ with "poses", "trans", "betas", "expressions", etc.
    motion_dict = np.load("/result_motion.npz", allow_pickle=True)

    # 2D face (freeze body, remove global motion)
    v2d_face = render2d(
        motion_dict,
        resolution=(512, 512),
        face_only=True,
        remove_global=True
    )
    write_video("/save_path_face_2d.mp4", v2d_face.permute(0, 2, 3, 1), fps=30)

    # 2D body (show entire body, keep global motion)
    v2d_body = render2d(
        motion_dict,
        resolution=(1080, 1920),
        face_only=False,
        remove_global=False
    )
    write_video("/save_path_body_2d.mp4", v2d_body.permute(0, 2, 3, 1), fps=30)

    # 3D face (freeze body, remove global motion)
    v3d_face = render3d(
        motion_dict,
        resolution=(512, 512),
        face_only=True,
        remove_global=True
    )
    write_video("/save_path_face_3d.mp4", v3d_face.permute(0, 2, 3, 1), fps=30)

    # 3D body (show entire body, keep global motion)
    v3d_body = render3d(
        motion_dict,
        resolution=(1080, 1920),
        face_only=False,
        remove_global=False
    )
    write_video("/save_path_body_3d.mp4", v3d_body.permute(0, 2, 3, 1), fps=30)

if __name__ == "__main__":
    example_usage()
