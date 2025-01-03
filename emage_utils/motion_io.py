import numpy as np
import torch
import smplx

MASK_DICT = {
    "local_upper": [
        False, False, False, True, False, False, True, False, False, True,
        False, False, True, True, True, True, True, True, True, True,
        True, True, False, False, False, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True
    ],
    "local_full": [False] + [True]*54
}

def select_with_mask(motion: np.ndarray, mask: list[bool]) -> np.ndarray:
    mask_arr = np.array(mask, dtype=bool)
    j = len(mask_arr)
    c_channels = motion.shape[-1] // j
    new_shape = motion.shape[:-1] + (j, c_channels)
    motion = motion.reshape(new_shape)
    selected_motion = motion[..., mask_arr, :]
    final_shape = selected_motion.shape[:-2] + (selected_motion.shape[-2]*selected_motion.shape[-1],)
    selected_motion = selected_motion.reshape(final_shape)
    return selected_motion

def recover_from_mask(selected_motion: np.ndarray, mask: list[bool]) -> np.ndarray:
    mask_arr = np.array(mask, dtype=bool)
    j = len(mask_arr)
    # Infer c_channels from selected_motion's last dimension and sum(mask)
    c_channels = selected_motion.shape[-1] // mask_arr.sum()
    new_shape = selected_motion.shape[:-1] + (mask_arr.sum(), c_channels)
    selected_motion = selected_motion.reshape(new_shape)
    out_shape = selected_motion.shape[:-2] + (j, c_channels)
    recovered = np.zeros(out_shape, dtype=selected_motion.dtype)
    recovered[..., mask_arr, :] = selected_motion
    final_shape = recovered.shape[:-2] + (j * c_channels,)
    recovered = recovered.reshape(final_shape)
    return recovered

def select_with_mask_ts(motion: torch.Tensor, mask: list[bool]) -> torch.Tensor:
    mask_arr = torch.tensor(mask, dtype=torch.bool, device=motion.device)
    j = len(mask_arr)
    c_channels = motion.shape[-1] // j
    new_shape = motion.shape[:-1] + (j, c_channels)
    motion = motion.reshape(new_shape)
    selected_motion = motion[..., mask_arr, :]
    final_shape = selected_motion.shape[:-2] + (selected_motion.shape[-2]*selected_motion.shape[-1],)
    selected_motion = selected_motion.reshape(final_shape)
    return selected_motion

def recover_from_mask_ts(selected_motion: torch.Tensor, mask: list[bool]) -> torch.Tensor:
    device = selected_motion.device
    dtype = selected_motion.dtype
    mask_arr = torch.tensor(mask, dtype=torch.bool, device=device)
    j = len(mask_arr)
    sum_mask = mask_arr.sum().item()
    c_channels = selected_motion.shape[-1] // sum_mask
    new_shape = selected_motion.shape[:-1] + (sum_mask, c_channels)
    selected_motion = selected_motion.reshape(new_shape)
    out_shape = list(selected_motion.shape[:-2]) + [j, c_channels]
    recovered = torch.zeros(out_shape, dtype=dtype, device=device)
    recovered[..., mask_arr, :] = selected_motion
    final_shape = list(recovered.shape[:-2]) + [j * c_channels]
    recovered = recovered.reshape(final_shape)
    return recovered

def time_upsample_numpy(data: np.ndarray, k: int) -> np.ndarray:
    # data: (..., t, c)
    # output: (..., k*t, c)
    if k == 1:
        return data.copy()
    shape = data.shape
    t = shape[-2]
    c = shape[-1]
    # original and new time indices
    original_t = np.arange(t)
    new_t = np.linspace(0, t - 1, k * t)

    # reshape to (M, c, t)
    reshaped = data.reshape(-1, t, c).transpose(0, 2, 1)
    M = reshaped.shape[0]
    reshaped = reshaped.reshape(M * c, t)

    # find interpolation indices
    idx = np.searchsorted(original_t, new_t, side='right') - 1
    idx = np.clip(idx, 0, t - 2)
    idx1 = idx + 1

    x0 = original_t[idx]
    x1 = original_t[idx1]
    w = (new_t - x0) / (x1 - x0)

    f0 = reshaped[:, idx]
    f1 = reshaped[:, idx1]

    out = f0 + (f1 - f0) * w
    out = out.reshape(M, c, k * t).transpose(0, 2, 1)
    final_shape = shape[:-2] + (k * t, c)
    return out.reshape(final_shape)

def beat_format_save(
    save_path: str,
    motion_data: np.ndarray,
    mask: list[bool] = None,
    betas: np.ndarray = None,
    expressions: np.ndarray = None,
    trans: np.ndarray = None,
    upsample: int = None,
):
    if betas is None:
        betas = np.zeros((motion_data.shape[0], 300), dtype=motion_data.dtype)
    if expressions is None:
        expressions = np.zeros((motion_data.shape[0], 100), dtype=motion_data.dtype)
    if trans is None:
        smplx_model = smplx.create(
            "./emage_evaltools/smplx_models/",
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False
        ).eval()
        betas_ts = torch.from_numpy(betas[0:1]).float()
        output = smplx_model(
            betas=betas_ts,
            transl=torch.zeros(1, 3),
            expression=torch.zeros(1, 100),
            jaw_pose=torch.zeros(1, 3),
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 63),
            left_hand_pose=torch.zeros(1, 45),
            right_hand_pose=torch.zeros(1, 45),
            return_joints=True,
            leye_pose=torch.zeros(1, 3),
            reye_pose=torch.zeros(1, 3)
        )
        trans = (output["joints"][:, 10, :] + output["joints"][:, 11, :]) / 2
        # print(trans)
        trans = -trans.repeat(motion_data.shape[0], 1).numpy()

    if mask is not None:
        motion_data = recover_from_mask(motion_data, mask)

    if upsample is not None and upsample > 1:
        motion_data = time_upsample_numpy(motion_data, upsample)
        betas = time_upsample_numpy(betas, upsample)
        expressions = time_upsample_numpy(expressions, upsample)
        trans = time_upsample_numpy(trans, upsample)

    np.savez(
        save_path,
        betas=betas[0],
        poses=motion_data,
        expressions=expressions,
        trans=trans,
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=30
    )

def beat_format_load(load_path: str, mask: list[bool] = None):
    data = np.load(load_path, allow_pickle=True)
    poses = data['poses']
    betas = data['betas']
    expressions = data['expressions']
    trans = data['trans']

    if mask is not None:
        poses = select_with_mask(poses, mask)

    return {
        "poses": poses,
        "betas": betas,
        "expressions": expressions,
        "trans": trans
    }