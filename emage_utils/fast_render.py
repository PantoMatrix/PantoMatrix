import os
import time
import numpy as np
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # or 'osmesa'
import pyrender
import trimesh
import queue
import imageio
import threading
import multiprocessing
import glob
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

args = {
    'render_video_fps': 30,
    'render_video_width': 480,
    'render_video_height': 720,
    'render_concurrent_num': max(1, multiprocessing.cpu_count() - 1)  ,
    'render_tmp_img_filetype': 'bmp',
    'debug': False
}

def deg_to_rad(degrees):
    return degrees * np.pi / 180

def create_pose_camera(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 1.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def create_pose_light(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def create_scene_with_mesh(vertices, faces, uniform_color, pose_camera, pose_light):
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=uniform_color)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=pose_camera)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(light, pose=pose_light)
    return scene

def do_render_one_frame(renderer, frame_idx, vertices, vertices1, faces):
    if frame_idx % 100 == 0:
        print('processed', frame_idx, 'frames')
    uniform_color = [220, 220, 220, 255]
    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = create_pose_light(angle_deg=-30)
    figs = []
    for vtx in [vertices, vertices1]:
        scene = create_scene_with_mesh(vtx, faces, uniform_color, pose_camera, pose_light)
        fig, _ = renderer.render(scene)
        figs.append(fig)
    return figs[0], figs[1]

def do_render_one_frame_no_gt(renderer, frame_idx, vertices, faces):
    if frame_idx % 100 == 0:
        print('processed', frame_idx, 'frames')
    uniform_color = [220, 220, 220, 255]
    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = create_pose_light(angle_deg=-30)
    scene = create_scene_with_mesh(vertices, faces, uniform_color, pose_camera, pose_light)
    fig, _ = renderer.render(scene)
    return fig

def write_images_from_queue(fig_queue, output_dir, img_filetype):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        fid, fig1, fig2 = e
        fn = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")
        merged_fig = np.hstack((fig1, fig2))
        try:
            imageio.imwrite(fn, merged_fig)
        except Exception as ex:
            print(f"Error writing image {fn}: {ex}")
            raise ex

def write_images_from_queue_no_gt(fig_queue, output_dir, img_filetype):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        fid, fig1 = e
        fn = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")
        try:
            imageio.imwrite(fn, fig1)
        except Exception as ex:
            print(f"Error writing image {fn}: {ex}")
            raise ex

def render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_width, render_height, fig_queue):
    fig_resolution = (render_width, render_height)
    renderer = pyrender.OffscreenRenderer(*fig_resolution)
    for idx, fid in enumerate(fids):
        fig1, fig2 = do_render_one_frame(renderer, fid, frame_vertex_pairs[idx][0], frame_vertex_pairs[idx][1], faces)
        fig_queue.put((fid, fig1, fig2))
    renderer.delete()

def render_frames_and_enqueue_no_gt(fids, frame_vertex_pairs, faces, render_width, render_height, fig_queue):
    fig_resolution = (render_width, render_height)
    renderer = pyrender.OffscreenRenderer(*fig_resolution)
    for idx, fid in enumerate(fids):
        fig1 = do_render_one_frame_no_gt(renderer, fid, frame_vertex_pairs[idx][0], faces)
        fig_queue.put((fid, fig1))
    renderer.delete()

def sub_process_process_frame(subprocess_index, render_video_width, render_video_height, render_tmp_img_filetype, fids, frame_vertex_pairs, faces, output_dir):
    t0 = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={t0}")
    fig_queue = queue.Queue()
    render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_video_width, render_video_height, fig_queue)
    fig_queue.put(None)
    t1 = time.time()
    thr = threading.Thread(target=write_images_from_queue, args=(fig_queue, output_dir, render_tmp_img_filetype))
    thr.start()
    thr.join()
    t2 = time.time()
    print(f"subprocess_index={subprocess_index} render={t1 - t0:.2f} all={t2 - t0:.2f}")

def sub_process_process_frame_no_gt(subprocess_index, render_video_width, render_video_height, render_tmp_img_filetype, fids, frame_vertex_pairs, faces, output_dir):
    t0 = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={t0}")
    fig_queue = queue.Queue()
    render_frames_and_enqueue_no_gt(fids, frame_vertex_pairs, faces, render_video_width, render_video_height, fig_queue)
    fig_queue.put(None)
    t1 = time.time()
    thr = threading.Thread(target=write_images_from_queue_no_gt, args=(fig_queue, output_dir, render_tmp_img_filetype))
    thr.start()
    thr.join()
    t2 = time.time()
    print(f"subprocess_index={subprocess_index} render={t1 - t0:.2f} all={t2 - t0:.2f}")

def distribute_frames(frames, vertices_all, vertices1_all):
    sample_interval = max(1, int(30 // args['render_video_fps']))
    subproc_frame_ids = [[] for _ in range(args['render_concurrent_num'])]
    subproc_vertices = [[] for _ in range(args['render_concurrent_num'])]
    sid = 0
    for i in range(frames):
        if i % sample_interval != 0:
            continue
        idx = sid % args['render_concurrent_num']
        subproc_frame_ids[idx].append(sid)
        subproc_vertices[idx].append((vertices_all[i], vertices1_all[i]))
        sid += 1
    return subproc_frame_ids, subproc_vertices

def distribute_frames_no_gt(frames, vertices_all):
    sample_interval = max(1, int(30 // args['render_video_fps']))
    subproc_frame_ids = [[] for _ in range(args['render_concurrent_num'])]
    subproc_vertices = [[] for _ in range(args['render_concurrent_num'])]
    sid = 0
    for i in range(frames):
        if i % sample_interval != 0:
            continue
        idx = sid % args['render_concurrent_num']
        subproc_frame_ids[idx].append(sid)
        subproc_vertices[idx].append((vertices_all[i], vertices_all[i]))
        sid += 1
    return subproc_frame_ids, subproc_vertices

def generate_silent_videos(frames, vertices_all, vertices1_all, faces, output_dir):
    ids, verts = distribute_frames(frames, vertices_all, vertices1_all)
    with multiprocessing.Pool(args['render_concurrent_num']) as pool:
        pool.starmap(sub_process_process_frame, [
            (
                i, 
                args['render_video_width'],
                args['render_video_height'],
                args['render_tmp_img_filetype'],
                ids[i],
                verts[i],
                faces,
                output_dir
            )
            for i in range(args['render_concurrent_num'])
        ])
    out_file = os.path.join(output_dir, "silence_video.mp4")
    convert_img_to_mp4(os.path.join(output_dir, f"frame_%d.{args['render_tmp_img_filetype']}"), out_file, args['render_video_fps'])
    for fn in glob.glob(os.path.join(output_dir, f"*.{args['render_tmp_img_filetype']}")):
        os.remove(fn)
    return out_file

def generate_silent_videos_no_gt(frames, vertices_all, faces, output_dir):
    ids, verts = distribute_frames_no_gt(frames, vertices_all)
    with multiprocessing.Pool(args['render_concurrent_num']) as pool:
        pool.starmap(sub_process_process_frame_no_gt, [
            (
                i, 
                args['render_video_width'],
                args['render_video_height'],
                args['render_tmp_img_filetype'],
                ids[i],
                verts[i],
                faces,
                output_dir
            )
            for i in range(args['render_concurrent_num'])
        ])
    out_file = os.path.join(output_dir, "silence_video.mp4")
    convert_img_to_mp4(os.path.join(output_dir, f"frame_%d.{args['render_tmp_img_filetype']}"), out_file, args['render_video_fps'])
    for fn in glob.glob(os.path.join(output_dir, f"*.{args['render_tmp_img_filetype']}")):
        os.remove(fn)
    return out_file

def add_audio_to_video(silent_video_path, audio_path, output_video_path):
    cmd = [
        'ffmpeg','-y','-i', silent_video_path,'-i', audio_path,'-map','0:v','-map','1:a','-c:v','copy','-shortest',output_video_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Video with audio generated: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def convert_img_to_mp4(input_pattern, output_file, framerate=30):
    cmd = ['ffmpeg','-framerate', str(framerate),'-i', input_pattern,'-c:v','libx264','-pix_fmt','yuv420p',output_file,'-y']
    try:
        subprocess.run(cmd, check=True)
        print(f"Video conversion: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def process_frame(i, vertices_all, vertices1_all, faces, output_dir, filenames):
    uniform_color = [220, 220, 220, 255]
    reso = (1000, 1000)
    fig, axs = plt.subplots(1, 2, figsize=(20,10))
    axs = axs.flatten()
    vertices = vertices_all[i]
    vertices1 = vertices1_all[i]
    fn = f"{output_dir}frame_{i}.png"
    if i % 100 == 0:
        print('processed', i, 'frames')
    angle_rad = deg_to_rad(-2)
    pose_camera = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 1.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    angle_rad = deg_to_rad(-30)
    pose_light = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    for idx, vtx in enumerate([vertices, vertices1]):
        tm = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=uniform_color)
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=True)
        scene = pyrender.Scene()
        scene.add(mesh)
        cam = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
        scene.add(cam, pose=pose_camera)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        scene.add(light, pose=pose_light)
        r = pyrender.OffscreenRenderer(*reso)
        color, _ = r.render(scene)
        axs[idx].imshow(color)
        axs[idx].axis('off')
        r.delete()
    plt.savefig(fn, bbox_inches='tight')
    plt.close(fig)

def generate_images(frames, vertices_all, vertices1_all, faces, output_dir, filenames):
    nc = multiprocessing.cpu_count() - 1
    for i in range(frames):
        process_frame(i*3, vertices_all, vertices1_all, faces, output_dir, filenames)

def render_one_sequence_with_face(res_npz_path, output_dir, audio_path, model_folder="/data/datasets/smplx_models/", model_type='smplx', gender='NEUTRAL_2020', ext='npz', num_betas=300, num_expression_coeffs=100, use_face_contour=False, use_matplotlib=False, remove_transl=True):
    import smplx
    import torch
    data_np_body = np.load(res_npz_path, allow_pickle=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    faces = np.load(f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]
    n = data_np_body["poses"].shape[0]
    model = smplx.create(model_folder, model_type=model_type, gender=gender, use_face_contour=use_face_contour, num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, ext=ext, use_pca=False).cuda()
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).cuda()
    jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).cuda()
    pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).cuda()
    transl = torch.from_numpy(data_np_body["trans"][:n]).to(torch.float32).cuda()
    if remove_transl: 
        transl = transl[0:1].repeat(n, 1)
    output = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose, global_orient=pose[:,:3], body_pose=pose[:,3:21*3+3], left_hand_pose=pose[:,25*3:40*3], right_hand_pose=pose[:,40*3:55*3], leye_pose=pose[:,69:72], reye_pose=pose[:,72:75], return_verts=True)
    vertices_all = output["vertices"].cpu().numpy()

    pose1 = torch.zeros_like(pose).to(torch.float32).cuda()
    output1 = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose, global_orient=pose1[:,:3], body_pose=pose1[:,3:21*3+3], left_hand_pose=pose1[:,25*3:40*3], right_hand_pose=pose1[:,40*3:55*3], leye_pose=pose1[:,69:72], reye_pose=pose1[:,72:75], return_verts=True)
    v1 = output1["vertices"].cpu().numpy()*7
    td = np.zeros_like(v1)
    td[:, :, 1] = 10
    vertices1_all = v1 - td
    if args['debug']:
        seconds = 1
    else:
        seconds = vertices_all.shape[0]//30
    sfile = generate_silent_videos(int(seconds*args['render_video_fps']), vertices1_all, vertices_all, faces, output_dir)
    base = os.path.splitext(os.path.basename(res_npz_path))[0]
    final_clip = os.path.join(output_dir, f"{base}.mp4")
    add_audio_to_video(sfile, audio_path, final_clip)
    os.remove(sfile)
    return final_clip

def render_one_sequence(res_npz_path, gt_npz_path, output_dir, audio_path, model_folder="/data/datasets/smplx_models/", model_type='smplx', gender='NEUTRAL_2020', ext='npz', num_betas=300, num_expression_coeffs=100, use_face_contour=False, use_matplotlib=False, remove_transl=True):
    import smplx
    import torch
    data_np_body = np.load(res_npz_path, allow_pickle=True)
    gt_np_body = np.load(gt_npz_path, allow_pickle=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    faces = np.load(f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]
    n = data_np_body["poses"].shape[0]
    model = smplx.create(model_folder, model_type=model_type, gender=gender, use_face_contour=use_face_contour, num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, ext=ext, use_pca=False).cuda()
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).cuda()
    jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).cuda()
    pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).cuda()
    transl = torch.from_numpy(data_np_body["trans"][:n]).to(torch.float32).cuda()
    if remove_transl: 
        transl = transl[0:1].repeat(n, 1)
    output = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose, global_orient=pose[:,:3], body_pose=pose[:,3:21*3+3], left_hand_pose=pose[:,25*3:40*3], right_hand_pose=pose[:,40*3:55*3], leye_pose=pose[:,69:72], reye_pose=pose[:,72:75], return_verts=True)
    vertices_all = output["vertices"].cpu().numpy()
    beta1 = torch.from_numpy(gt_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    expression1 = torch.from_numpy(gt_np_body["expressions"][:n]).to(torch.float32).cuda()
    jaw_pose1 = torch.from_numpy(gt_np_body["poses"][:n,66:69]).to(torch.float32).cuda()
    pose1 = torch.from_numpy(gt_np_body["poses"][:n]).to(torch.float32).cuda()
    transl1 = torch.from_numpy(gt_np_body["trans"][:n]).to(torch.float32).cuda()
    if remove_transl: 
        transl1 = transl1[0:1].repeat(n, 1)
    output1 = model(betas=beta1, transl=transl1, expression=expression1, jaw_pose=jaw_pose1, global_orient=pose1[:,:3], body_pose=pose1[:,3:21*3+3], left_hand_pose=pose1[:,25*3:40*3], right_hand_pose=pose1[:,40*3:55*3], leye_pose=pose1[:,69:72], reye_pose=pose1[:,72:75], return_verts=True)
    vertices1_all = output1["vertices"].cpu().numpy()
    if args['debug']:
        seconds = 1
    else:
        seconds = vertices_all.shape[0]//30
    sfile = generate_silent_videos(int(seconds*args['render_video_fps']), vertices_all, vertices1_all, faces, output_dir)
    base = os.path.splitext(os.path.basename(res_npz_path))[0]
    final_clip = os.path.join(output_dir, f"{base}.mp4")
    add_audio_to_video(sfile, audio_path, final_clip)
    os.remove(sfile)
    return final_clip

def render_one_sequence_no_gt(res_npz_path, output_dir, audio_path, model_folder="/data/datasets/smplx_models/", model_type='smplx', gender='NEUTRAL_2020', ext='npz', num_betas=300, num_expression_coeffs=100, use_face_contour=False, use_matplotlib=False, remove_transl=True):
    import smplx
    import torch
    data_np_body = np.load(res_npz_path, allow_pickle=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    faces = np.load(f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]
    n = data_np_body["poses"].shape[0]
    model = smplx.create(model_folder, model_type=model_type, gender=gender, use_face_contour=use_face_contour, num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, ext=ext, use_pca=False).cuda()
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).cuda()
    jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).cuda()
    pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).cuda()
    transl = torch.from_numpy(data_np_body["trans"][:n]).to(torch.float32).cuda()
    if remove_transl: 
        transl = transl[0:1].repeat(n, 1)
    output = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose, global_orient=pose[:,:3], body_pose=pose[:,3:21*3+3], left_hand_pose=pose[:,25*3:40*3], right_hand_pose=pose[:,40*3:55*3], leye_pose=pose[:,69:72], reye_pose=pose[:,72:75], return_verts=True)
    vertices_all = output["vertices"].cpu().numpy()
    if args['debug']:
        seconds = 1
    else:
        seconds = vertices_all.shape[0]//30
    sfile = generate_silent_videos_no_gt(int(seconds*args['render_video_fps']), vertices_all, faces, output_dir)
    base = os.path.splitext(os.path.basename(res_npz_path))[0]
    final_clip = os.path.join(output_dir, f"{base}.mp4")
    add_audio_to_video(sfile, audio_path, final_clip)
    os.remove(sfile)
    return final_clip

def render_one_sequence_face_only(res_npz_path, output_dir, audio_path, model_folder="/data/datasets/smplx_models/", model_type='smplx', gender='NEUTRAL_2020', ext='npz', num_betas=300, num_expression_coeffs=100, use_face_contour=False, use_matplotlib=False, remove_transl=True):
    import smplx
    import torch
    data_np_body = np.load(res_npz_path, allow_pickle=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    faces = np.load(f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]
    n = data_np_body["poses"].shape[0]
    model = smplx.create(model_folder, model_type=model_type, gender=gender, use_face_contour=use_face_contour, num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, ext=ext, use_pca=False).cuda()
    beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).cuda()
    beta = beta.repeat(n, 1)
    expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).cuda()
    jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).cuda()
    pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).cuda()
    transl = torch.from_numpy(data_np_body["trans"][:n]).to(torch.float32).cuda()
    if remove_transl: 
        transl = transl[0:1].repeat(n, 1)
    output = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose, global_orient=pose[:,:3], body_pose=pose[:,3:21*3+3], left_hand_pose=pose[:,25*3:40*3], right_hand_pose=pose[:,40*3:55*3], leye_pose=pose[:,69:72], reye_pose=pose[:,72:75], return_verts=True)
    vertices_all = output["vertices"].cpu().numpy()

    pose1 = torch.zeros_like(pose).to(torch.float32).cuda()
    output1 = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose, global_orient=pose1[:,:3], body_pose=pose1[:,3:21*3+3], left_hand_pose=pose1[:,25*3:40*3], right_hand_pose=pose1[:,40*3:55*3], leye_pose=pose1[:,69:72], reye_pose=pose1[:,72:75], return_verts=True)
    v1 = output1["vertices"].cpu().numpy()*7
    td = np.zeros_like(v1)
    td[:, :, 1] = 10
    vertices_all = v1 - td

    if args['debug']:
        seconds = 1
    else:
        seconds = vertices_all.shape[0]//30
    sfile = generate_silent_videos_no_gt(int(seconds*args['render_video_fps']), vertices_all, faces, output_dir)
    base = os.path.splitext(os.path.basename(res_npz_path))[0]
    final_clip = os.path.join(output_dir, f"{base}.mp4")
    add_audio_to_video(sfile, audio_path, final_clip)
    os.remove(sfile)
    return final_clip