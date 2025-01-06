import os
import argparse
import torch
from torchvision.io import write_video
import librosa
import time
import numpy as np
from tqdm import tqdm
from emage_utils.motion_io import beat_format_save
from emage_utils import fast_render
from models.disco_audio import DiscoAudioModel


def inference(model, audio_path, device, save_folder, sr, pose_fps, seed_frames):
    audio, _ = librosa.load(audio_path, sr=sr)
    audio = torch.from_numpy(audio).to(device).unsqueeze(0)
    speaker_id = torch.zeros(1,1).long().to(device)
    with torch.no_grad():
        motion_pred = model(audio, speaker_id, seed_frames=seed_frames, seed_motion=None)["motion_axis_angle"]
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
    beat_format_save(os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz"),
                     motion_pred, upsample=30//pose_fps)
    return t


def visualize_one(save_folder, audio_path, nopytorch3d=False):
    npz_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz")
    motion_dict = np.load(npz_path, allow_pickle=True)
    if not nopytorch3d:
        from emage_utils.npz2pose import render2d
        v2d_body = render2d(motion_dict, (720, 480), face_only=False, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dbody.mp4"), v2d_body.permute(0, 2, 3, 1), fps=30)
        fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dbody.mp4"), audio_path, npz_path.replace(".npz", "_2dbody_audio.mp4"))
    fast_render.render_one_sequence_no_gt(npz_path, os.path.dirname(npz_path), audio_path, model_folder="./emage_evaltools/smplx_models/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", type=str, default="./examples/audio")
    parser.add_argument("--save_folder", type=str, default="./examples/motion")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--nopytorch3d", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiscoAudioModel.from_pretrained("H-Liu1997/disco_audio").to(device)
    model.eval()

    audio_files = [os.path.join(args.audio_folder, f) for f in os.listdir(args.audio_folder) if f.endswith(".wav")]
    sr, pose_fps, seed_frames = model.cfg.audio_sr, model.cfg.pose_fps, model.cfg.seed_frames
    all_t = 0
    start_time = time.time()
    for audio_path in tqdm(audio_files, desc="Inference"):
        all_t += inference(model, audio_path, device, args.save_folder, sr, pose_fps, seed_frames)
    print(f"generate total {all_t/pose_fps:.2f} seconds motion in {time.time()-start_time:.2f} seconds, saved in {args.save_folder}")
    
    start_time = time.time()
    if args.visualization:
        for audio_path in tqdm(audio_files, desc="Visualize"):
            visualize_one(args.save_folder, audio_path, args.nopytorch3d)
        print(f"render total {all_t/pose_fps:.2f} seconds motion in {time.time()-start_time:.2f} seconds, saved in {args.save_folder}")


if __name__ == "__main__":
    main()