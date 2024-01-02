import os
import numpy as np
import random
import torch
import shutil
import csv
import pprint
import pandas as pd
from loguru import logger
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle
import time

import numpy as np

def adjust_array(x, k):
    len_x = len(x)
    len_k = len(k)

    # If x is shorter than k, pad with zeros
    if len_x < len_k:
        return np.pad(x, (0, len_k - len_x), 'constant')

    # If x is longer than k, truncate x
    elif len_x > len_k:
        return x[:len_k]

    # If both are of same length
    else:
        return x

def onset_to_frame(onset_times, audio_length, fps):
    # Calculate total number of frames for the given audio length
    total_frames = int(audio_length * fps)
    
    # Create an array of zeros of shape (total_frames,)
    frame_array = np.zeros(total_frames, dtype=np.int32)
    
    # For each onset time, calculate the frame number and set it to 1
    for onset in onset_times:
        frame_num = int(onset * fps)
        # Check if the frame number is within the array bounds
        if 0 <= frame_num < total_frames:
            frame_array[frame_num] = 1
    
    return frame_array

def smooth_animations(animation1, animation2, blend_frames):
    """
    Smoothly transition between two animation clips using linear interpolation.

    Parameters:
    - animation1: The first animation clip, a numpy array of shape [n, k].
    - animation2: The second animation clip, a numpy array of shape [n, k].
    - blend_frames: Number of frames over which to blend the two animations.

    Returns:
    - A smoothly blended animation clip of shape [2n, k].
    """
    
    # Ensure blend_frames doesn't exceed the length of either animation
    blend_frames = min(blend_frames, len(animation1), len(animation2))
    
    # Extract overlapping sections
    overlap_a1 = animation1[-blend_frames:-blend_frames+1, :]
    overlap_a2 = animation2[blend_frames-1:blend_frames, :]
    
    # Create blend weights for linear interpolation
    alpha = np.linspace(0, 1, 2 * blend_frames).reshape(-1, 1)
    
    # Linearly interpolate between overlapping sections
    blended_overlap = overlap_a1 * (1 - alpha) + overlap_a2 * alpha
    
    # Extend the animations to form the result with 2n frames
    if blend_frames == len(animation1) and blend_frames == len(animation2):
        result = blended_overlap
    else:
        before_blend = animation1[:-blend_frames]
        after_blend = animation2[blend_frames:]
        result = np.vstack((before_blend, blended_overlap, after_blend))
    return result


def interpolate_sequence(quaternions):
    bs, n, j, _ = quaternions.shape
    new_n = 2 * n
    new_quaternions = torch.zeros((bs, new_n, j, 4), device=quaternions.device, dtype=quaternions.dtype)

    for i in range(n):
        q1 = quaternions[:, i, :, :]
        new_quaternions[:, 2*i, :, :] = q1

        if i < n - 1:
            q2 = quaternions[:, i + 1, :, :]
            new_quaternions[:, 2*i + 1, :, :] = slerp(q1, q2, 0.5)
        else:
            # For the last point, duplicate the value
            new_quaternions[:, 2*i + 1, :, :] = q1

    return new_quaternions

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def quaternion_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def slerp(q1, q2, t):
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    flip = (dot < 0).float()
    q2 = (1 - flip * 2) * q2
    dot = dot * (1 - flip * 2)

    DOT_THRESHOLD = 0.9995
    mask = (dot > DOT_THRESHOLD).float()

    theta_0 = torch.acos(dot)
    theta = theta_0 * t

    q3 = q2 - q1 * dot
    q3 = q3 / torch.norm(q3, dim=-1, keepdim=True)

    interpolated = (torch.cos(theta) * q1 + torch.sin(theta) * q3)

    return mask * (q1 + t * (q2 - q1)) + (1 - mask) * interpolated

def estimate_linear_velocity(data_seq, dt):
    '''
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order
    differences, respectively
    - h : step size
    '''
    # first steps is forward diff (t+1 - t) / dt
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    # middle steps are second order (t+1 - t-1) / 2dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
    # last step is backward diff (t - t-1) / dt
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq


def estimate_angular_velocity(rot_seq, dt):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_linear_velocity(rot_seq, dt)
    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector by averaging symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w

import matplotlib.image as mpimg
from io import BytesIO

def image_from_bytes(image_bytes):
    return mpimg.imread(BytesIO(image_bytes), format='PNG')



def process_frame(i, vertices_all, vertices1_all, faces, output_dir, use_matplotlib, filenames, camera_params, camera_params1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    import pyvirtualdisplay as Display

    vertices = vertices_all[i]
    vertices1 = vertices1_all[i]
    filename = f"{output_dir}frame_{i}.png"
    filenames.append(filename)
    if i%100 == 0:
        print('processed', i, 'frames')
    #time_s = time.time()
    #print(vertices.shape)
    if use_matplotlib:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(121, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #ax.view_init(elev=0, azim=90)
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        ax.scatter(x, y, z, s=0.5)
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-0.5, 1.5])#heigth
        ax.set_zlim([-0, 2])#depth
        ax.set_box_aspect((1,1,1))
    else:
        mesh = trimesh.Trimesh(vertices, faces)
        scene = mesh.scene()
        scene.camera.fov = camera_params['fov']
        scene.camera.resolution = camera_params['resolution']
        scene.camera.z_near = camera_params['z_near']
        scene.camera.z_far = camera_params['z_far']
        scene.graph[scene.camera.name] = camera_params['transform']
        fig, ax =plt.subplots(1,2, figsize=(16, 6))
        image = scene.save_image(resolution=[640, 480], visible=False)  
        im0 = ax[0].imshow(image_from_bytes(image))
        ax[0].axis('off')
    
    if use_matplotlib:
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_box_aspect((1,1,1))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        x1 = vertices1[:, 0]
        y1 = vertices1[:, 1]
        z1 = vertices1[:, 2]
        ax2.scatter(x1, y1, z1, s=0.5)
        ax2.set_xlim([-1.0, 1.0])
        ax2.set_ylim([-0.5, 1.5])#heigth
        ax2.set_zlim([-0, 2])
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        mesh1 = trimesh.Trimesh(vertices1, faces)
        scene1 = mesh1.scene()
        scene1.camera.fov = camera_params1['fov']
        scene1.camera.resolution = camera_params1['resolution']
        scene1.camera.z_near = camera_params1['z_near']
        scene1.camera.z_far = camera_params1['z_far']
        scene1.graph[scene1.camera.name] = camera_params1['transform']
        image1 = scene1.save_image(resolution=[640, 480], visible=False)
        im1 = ax[1].imshow(image_from_bytes(image1))
        ax[1].axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)    

def generate_images(frames, vertices_all, vertices1_all, faces, output_dir, use_matplotlib, filenames):
    import multiprocessing
    import trimesh
    num_cores = multiprocessing.cpu_count()  # This will get the number of cores on your machine.
    mesh = trimesh.Trimesh(vertices_all[0], faces)
    scene = mesh.scene()
    camera_params = {
        'fov': scene.camera.fov,
        'resolution': scene.camera.resolution,
        'focal': scene.camera.focal,
        'z_near': scene.camera.z_near,
        "z_far": scene.camera.z_far,
        'transform': scene.graph[scene.camera.name][0]
    }
    mesh1 = trimesh.Trimesh(vertices1_all[0], faces)
    scene1 = mesh1.scene()
    camera_params1 = {
        'fov': scene1.camera.fov,
        'resolution': scene1.camera.resolution,
        'focal': scene1.camera.focal,
        'z_near': scene1.camera.z_near,
        "z_far": scene1.camera.z_far,
        'transform': scene1.graph[scene1.camera.name][0]
    }
    # Use a Pool to manage the processes
    # print(num_cores)
    progress = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    with multiprocessing.Pool(num_cores) as pool:
        pool.starmap(process_frame, [(i, vertices_all, vertices1_all, faces, output_dir, use_matplotlib, filenames, camera_params, camera_params1) for i in range(frames)])

def render_one_sequence(
         res_npz_path,
         gt_npz_path,
         output_dir,
         audio_path,
         model_folder="/data/datasets/smplx_models/",
         model_type='smplx',
         gender='NEUTRAL_2020',
         ext='npz',
         num_betas=300,
         num_expression_coeffs=100,
         use_face_contour=False,
         use_matplotlib=False,
         args=None):
    import smplx
    import matplotlib.pyplot as plt
    import imageio
    from tqdm import tqdm
    import os
    import numpy as np 
    import torch
    import moviepy.editor as mp
    import librosa
    
    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext, use_pca=False).cuda()
    
    #data_npz = np.load(f"{output_dir}{res_npz_path}.npz")
    data_np_body = np.load(res_npz_path, allow_pickle=True)
    gt_np_body = np.load(gt_npz_path, allow_pickle=True)
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filenames = []
    if not use_matplotlib:
       import trimesh 
       #import pyrender
       from pyvirtualdisplay import Display
       display = Display(visible=0, size=(640, 480))
       display.start()
       faces = np.load(f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]     
    seconds = 1
    #data_npz["jaw_pose"].shape[0]
    n = data_np_body["poses"].shape[0]
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).cuda()
    jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).cuda()
    pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).cuda()
    transl = torch.from_numpy(data_np_body["trans"][:n]).to(torch.float32).cuda()
    # print(beta.shape, expression.shape, jaw_pose.shape, pose.shape, transl.shape, pose[:,:3].shape)
    output = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose,
        global_orient=pose[:,:3], body_pose=pose[:,3:21*3+3], left_hand_pose=pose[:,25*3:40*3], right_hand_pose=pose[:,40*3:55*3],
        leye_pose=pose[:, 69:72], 
        reye_pose=pose[:, 72:75],
        return_verts=True)
    vertices_all = output["vertices"].cpu().detach().numpy()

    beta1 = torch.from_numpy(gt_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    expression1 = torch.from_numpy(gt_np_body["expressions"][:n]).to(torch.float32).cuda()
    jaw_pose1 = torch.from_numpy(gt_np_body["poses"][:n,66:69]).to(torch.float32).cuda()
    pose1 = torch.from_numpy(gt_np_body["poses"][:n]).to(torch.float32).cuda()
    transl1 = torch.from_numpy(gt_np_body["trans"][:n]).to(torch.float32).cuda()
    output1 = model(betas=beta1, transl=transl1, expression=expression1, jaw_pose=jaw_pose1, global_orient=pose1[:,:3], body_pose=pose1[:,3:21*3+3], left_hand_pose=pose1[:,25*3:40*3], right_hand_pose=pose1[:,40*3:55*3],      
        leye_pose=pose1[:, 69:72], 
        reye_pose=pose1[:, 72:75],return_verts=True)
    vertices1_all = output1["vertices"].cpu().detach().numpy()
    if args.debug:
        seconds = 1
    else:
        seconds = vertices_all.shape[0]//30
    # camera_settings = None    
    time_s = time.time()
    generate_images(int(seconds*30), vertices_all, vertices1_all, faces, output_dir, use_matplotlib, filenames)
    filenames = [f"{output_dir}frame_{i}.png" for i in range(int(seconds*30))]  
    # print(time.time()-time_s)
    # for i in tqdm(range(seconds*30)):
    #     vertices = vertices_all[i]
    #     vertices1 = vertices1_all[i]
    #     filename = f"{output_dir}frame_{i}.png"
    #     filenames.append(filename)
    #     #time_s = time.time()
    #     #print(vertices.shape)
    #     if use_matplotlib:
    #         fig = plt.figure(figsize=(20, 10))
    #         ax = fig.add_subplot(121, projection="3d")
    #         fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #         #ax.view_init(elev=0, azim=90)
    #         x = vertices[:, 0]
    #         y = vertices[:, 1]
    #         z = vertices[:, 2]
    #         ax.scatter(x, y, z, s=0.5)
    #         ax.set_xlim([-1.0, 1.0])
    #         ax.set_ylim([-0.5, 1.5])#heigth
    #         ax.set_zlim([-0, 2])#depth
    #         ax.set_box_aspect((1,1,1))
    #     else:
    #         mesh = trimesh.Trimesh(vertices, faces)
    #         if i == 0:
    #             scene = mesh.scene()
    #             camera_params = {
    #                 'fov': scene.camera.fov,
    #                 'resolution': scene.camera.resolution,
    #                 'focal': scene.camera.focal,
    #                 'z_near': scene.camera.z_near,
    #                 "z_far": scene.camera.z_far,
    #                 'transform': scene.graph[scene.camera.name][0]
    #             }
    #         else: 
    #             scene = mesh.scene()
    #             scene.camera.fov = camera_params['fov']
    #             scene.camera.resolution = camera_params['resolution']
    #             scene.camera.z_near = camera_params['z_near']
    #             scene.camera.z_far = camera_params['z_far']
    #             scene.graph[scene.camera.name] = camera_params['transform']
    #         fig, ax =plt.subplots(1,2, figsize=(16, 6))
    #         image = scene.save_image(resolution=[640, 480], visible=False)
    #         #print((time.time()-time_s))   
    #         im0 = ax[0].imshow(image_from_bytes(image))
    #         ax[0].axis('off')

    #     # beta1 = torch.from_numpy(gt_np_body["betas"]).to(torch.float32).unsqueeze(0)
    #     # expression1 = torch.from_numpy(gt_np_body["expressions"][i]).to(torch.float32).unsqueeze(0)
    #     # jaw_pose1 = torch.from_numpy(gt_np_body["poses"][i][66:69]).to(torch.float32).unsqueeze(0)
    #     # pose1 = torch.from_numpy(gt_np_body["poses"][i]).to(torch.float32).unsqueeze(0)
    #     # transl1 = torch.from_numpy(gt_np_body["trans"][i]).to(torch.float32).unsqueeze(0)
    #     # #print(beta.shape, expression.shape, jaw_pose.shape, pose.shape, transl.shape)global_orient=pose[0:1,:3],
    #     # output1 = model(betas=beta1, transl=transl1, expression=expression1, jaw_pose=jaw_pose1, global_orient=pose1[0:1,:3], body_pose=pose1[0:1,3:21*3+3], left_hand_pose=pose1[0:1,25*3:40*3], right_hand_pose=pose1[0:1,40*3:55*3], return_verts=True)
    #     # vertices1 = output1["vertices"].cpu().detach().numpy()[0]
        
    #     if use_matplotlib:
    #         ax2 = fig.add_subplot(122, projection="3d")
    #         ax2.set_box_aspect((1,1,1))
    #         fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #         #ax2.view_init(elev=0, azim=90)
    #         x1 = vertices1[:, 0]
    #         y1 = vertices1[:, 1]
    #         z1 = vertices1[:, 2]
    #         ax2.scatter(x1, y1, z1, s=0.5)
    #         ax2.set_xlim([-1.0, 1.0])
    #         ax2.set_ylim([-0.5, 1.5])#heigth
    #         ax2.set_zlim([-0, 2])
    #         plt.savefig(filename, bbox_inches='tight')
    #         plt.close(fig)
    #     else:
    #         mesh1 = trimesh.Trimesh(vertices1, faces)
    #         if i == 0:
    #             scene1 = mesh1.scene()
    #             camera_params1 = {
    #                 'fov': scene1.camera.fov,
    #                 'resolution': scene1.camera.resolution,
    #                 'focal': scene1.camera.focal,
    #                 'z_near': scene1.camera.z_near,
    #                 "z_far": scene1.camera.z_far,
    #                 'transform': scene1.graph[scene1.camera.name][0]
    #             }
    #         else: 
    #             scene1 = mesh1.scene()
    #             scene1.camera.fov = camera_params1['fov']
    #             scene1.camera.resolution = camera_params1['resolution']
    #             scene1.camera.z_near = camera_params1['z_near']
    #             scene1.camera.z_far = camera_params1['z_far']
    #             scene1.graph[scene1.camera.name] = camera_params1['transform']
    #         image1 = scene1.save_image(resolution=[640, 480], visible=False)
    #         im1 = ax[1].imshow(image_from_bytes(image1))
    #         ax[1].axis('off')
    #         plt.savefig(filename, bbox_inches='tight')
    #         plt.close(fig)

    # display.stop()
    # print(filenames)
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(f"{output_dir}raw_{res_npz_path.split('/')[-1][:-4]}.mp4", images, fps=30)
    for filename in filenames:
        os.remove(filename)
        
    video = mp.VideoFileClip(f"{output_dir}raw_{res_npz_path.split('/')[-1][:-4]}.mp4")
    # audio, sr = librosa.load(audio_path)
    # audio = audio[:seconds*sr]
    # print(audio.shape, seconds, sr)
    # import soundfile as sf
    # sf.write(f"{output_dir}{res_npz_path.split('/')[-1][:-4]}.wav", audio, 16000, 'PCM_24')
    # audio_tmp = librosa.output.write_wav(f"{output_dir}{res_npz_path.split('/')[-1][:-4]}.wav", audio, sr=16000)
    audio = mp.AudioFileClip(audio_path)
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(f"{output_dir}{res_npz_path.split('/')[-1][4:-4]}.mp4")
    os.remove(f"{output_dir}raw_{res_npz_path.split('/')[-1][:-4]}.mp4")

def print_exp_info(args):
    logger.info(pprint.pformat(vars(args)))
    logger.info(f"# ------------ {args.name} ----------- #")
    logger.info("PyTorch version: {}".format(torch.__version__))
    logger.info("CUDA version: {}".format(torch.version.cuda))
    logger.info("{} GPUs".format(torch.cuda.device_count()))
    logger.info(f"Random Seed: {args.random_seed}")

def args2csv(args, get_head=False, list4print=[]):
    for k, v in args.items():
        if isinstance(args[k], dict):
            args2csv(args[k], get_head, list4print)
        else: list4print.append(k) if get_head else list4print.append(v)
    return list4print

class EpochTracker:
    def __init__(self, metric_names, metric_directions):
        assert len(metric_names) == len(metric_directions), "Metric names and directions should have the same length"


        self.metric_names = metric_names
        self.states = ['train', 'val', 'test']
        self.types = ['last', 'best']


        self.values = {name: {state: {type_: {'value': np.inf if not is_higher_better else -np.inf, 'epoch': 0}
                                       for type_ in self.types}
                              for state in self.states}
                      for name, is_higher_better in zip(metric_names, metric_directions)}
                     
        self.loss_meters = {name: {state: AverageMeter(f"{name}_{state}")
                                   for state in self.states}
                            for name in metric_names}


        self.is_higher_better = {name: direction for name, direction in zip(metric_names, metric_directions)}
        self.train_history = {name: [] for name in metric_names}
        self.val_history = {name: [] for name in metric_names}


    def update_meter(self, name, state, value):
        self.loss_meters[name][state].update(value)


    def update_values(self, name, state, epoch):
        value_avg = self.loss_meters[name][state].avg
        new_best = False


        if ((value_avg < self.values[name][state]['best']['value'] and not self.is_higher_better[name]) or
           (value_avg > self.values[name][state]['best']['value'] and self.is_higher_better[name])):
            self.values[name][state]['best']['value'] = value_avg
            self.values[name][state]['best']['epoch'] = epoch
            new_best = True
        self.values[name][state]['last']['value'] = value_avg
        self.values[name][state]['last']['epoch'] = epoch
        return new_best


    def get(self, name, state, type_):
        return self.values[name][state][type_]


    def reset(self):
        for name in self.metric_names:
            for state in self.states:
                self.loss_meters[name][state].reset()


    def flatten_values(self):
        flat_dict = {}
        for name in self.metric_names:
            for state in self.states:
                for type_ in self.types:
                    value_key = f"{name}_{state}_{type_}"
                    epoch_key = f"{name}_{state}_{type_}_epoch"
                    flat_dict[value_key] = self.values[name][state][type_]['value']
                    flat_dict[epoch_key] = self.values[name][state][type_]['epoch']
        return flat_dict
   
    def update_and_plot(self, name, epoch, save_path):
        new_best_train = self.update_values(name, 'train', epoch)
        new_best_val = self.update_values(name, 'val', epoch)


        self.train_history[name].append(self.loss_meters[name]['train'].avg)
        self.val_history[name].append(self.loss_meters[name]['val'].avg)


        train_values = self.train_history[name]
        val_values = self.val_history[name]
        epochs = list(range(1, len(train_values) + 1))


        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_values, label='Train')
        plt.plot(epochs, val_values, label='Val')
        plt.title(f'Train vs Val {name} over epochs')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()
        plt.savefig(save_path)
        plt.close()


        return new_best_train, new_best_val




def record_trial(args, tracker):
    """
    1. record notes, score, env_name, experments_path,
    """
    csv_path = args.out_path + "custom/" +args.csv_name+".csv"
    all_print_dict = vars(args)
    all_print_dict.update(tracker.flatten_values())
    if not os.path.exists(csv_path):
        pd.DataFrame([all_print_dict]).to_csv(csv_path, index=False)
    else:
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame([all_print_dict])
        df_aligned = df_existing.append(df_new).fillna("")
        df_aligned.to_csv(csv_path, index=False)
        

def set_random_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.deterministic #args.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.enabled = args.cudnn_enabled
    

def save_checkpoints(save_path, model, opt=None, epoch=None, lrs=None):
    if lrs is not None:
        states = { 'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'opt_state': opt.state_dict(),
                'lrs':lrs.state_dict(),}
    elif opt is not None:
        states = { 'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'opt_state': opt.state_dict(),}
    else:
        states = { 'model_state': model.state_dict(),}
    torch.save(states, save_path)


def load_checkpoints(model, save_path, load_name='model'):
    states = torch.load(save_path)
    new_weights = OrderedDict()
    flag=False
    for k, v in states['model_state'].items():
        #print(k)
        if "module" not in k:
            break
        else:
            new_weights[k[7:]]=v
            flag=True
    if flag: 
        try:
            model.load_state_dict(new_weights)
        except:
            #print(states['model_state'])
            model.load_state_dict(states['model_state'])
    else:
        model.load_state_dict(states['model_state'])
    logger.info(f"load self-pretrained checkpoints for {load_name}")


def model_complexity(model, args):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model,  (args.T_GLOBAL._DIM, args.TRAIN.CROP, args.TRAIN), 
        as_strings=False, print_per_layer_stat=False)
    logging.info('{:<30}  {:<8} BFlops'.format('Computational complexity: ', flops / 1e9))
    logging.info('{:<30}  {:<8} MParams'.format('Number of parameters: ', params / 1e6))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)