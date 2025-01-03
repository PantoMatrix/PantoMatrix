<div align="center">
<h1>PantoMatrix: Talking Face and Body Animation Generation</h1> 
PantoMatrix is an Open-Source and research project to generate 3D body and face animation from speech. It works as an API inputs speech audio and outputs body and face motion parameters. You may transfer these motion parameters to other formats such as Iphone ARKit Blendshape Weights or Vicon Skeleton bvh files based on your needs. 
<br>
<br>
</div>
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/emage-towards-unified-holistic-co-speech/3d-face-animation-on-beat2)](https://paperswithcode.com/sota/3d-face-animation-on-beat2?p=emage-towards-unified-holistic-co-speech)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/emage-towards-unified-holistic-co-speech/gesture-generation-on-beat2)](https://paperswithcode.com/sota/gesture-generation-on-beat2?p=emage-towards-unified-holistic-co-speech) -->

<div align="center">
  <img src="assets/intro.gif" alt="Animation Example" style="width:95%; clip-path: inset(0px 0px 3px 0px);">
</div>

<br>

<div align="center">
    <a href="https://pantomatrix.github.io/EMAGE/"><img src="https://img.shields.io/badge/Project-EMAGE-skyblue?logo=github&amp"></a>
    <!-- <a href="https://github.com/PantoMatrix/PantoMatrix/blob/main/scripts/EMAGE_2024/readme.md"><img src="https://img.shields.io/badge/Readme-gray?logo=readthedocs&amp"></a> -->
    <a href="https://www.youtube.com/watch?v=T0OYPvViFGE"><img src="https://img.shields.io/badge/Youtube-gray?logo=youtube&amp"></a>
    <a href="https://replicate.com/camenduru/emage"><img src="https://img.shields.io/badge/Replicate-gray?logo=google&amp"></a>
    <a href="https://colab.research.google.com/drive/1bB3LqAzceNTW2urXeMpOTPoJYTRoKlzB?usp=sharing"><img src="https://img.shields.io/badge/Colab-gray?logo=Google%20Colab&amp"></a>
    <a href="https://huggingface.co/spaces/H-Liu1997/EMAGE"><img src="https://img.shields.io/badge/Huggingface Space-gray?logo=huggingface&amp"></a>
    <a href="https://www.youtube.com/watch?v=T0OYPvViFGE"><img src="https://img.shields.io/badge/CVPR 2024-gray?logo=arxiv&amp"></a>
</div>

<div align="center">
    <a href="https://pantomatrix.github.io/BEAT/"><img src="https://img.shields.io/badge/Project-BEAT-lightyellow?logo=github&amp"></a>
    <!-- <a href="https://github.com/PantoMatrix/PantoMatrix/blob/main/scripts/EMAGE_2024/readme.md"><img src="https://img.shields.io/badge/Readme-gray?logo=readthedocs&amp"></a> -->
    <a href="https://www.youtube.com/watch?v=T0OYPvViFGE"><img src="https://img.shields.io/badge/Youtube-gray?logo=youtube&amp"></a>
    <!-- <a href="https://replicate.com/camenduru/emage"><img src="https://img.shields.io/badge/Replicate-gray?logo=google&amp"></a> -->
    <a href="https://colab.research.google.com/drive/1bB3LqAzceNTW2urXeMpOTPoJYTRoKlzB?usp=sharing"><img src="https://img.shields.io/badge/Colab-gray?logo=Google%20Colab&amp"></a>
    <!-- <a href="https://huggingface.co/spaces/H-Liu1997/EMAGE"><img src="https://img.shields.io/badge/Huggingface Space-gray?logo=huggingface&amp"></a> -->
    <a href="https://www.youtube.com/watch?v=T0OYPvViFGE"><img src="https://img.shields.io/badge/ECCV 2022-gray?logo=arxiv&amp"></a>
</div>

<div align="center">
    <a href="https://pantomatrix.github.io/DisCo/"><img src="https://img.shields.io/badge/Project-DisCo-pink?logo=github&amp"></a>
    <!-- <a href="https://github.com/PantoMatrix/PantoMatrix/blob/main/scripts/EMAGE_2024/readme.md"><img src="https://img.shields.io/badge/Readme-gray?logo=readthedocs&amp"></a> -->
    <a href="https://www.youtube.com/watch?v=T0OYPvViFGE"><img src="https://img.shields.io/badge/Youtube-gray?logo=youtube&amp"></a>
    <!-- <a href="https://replicate.com/camenduru/emage"><img src="https://img.shields.io/badge/Replicate-gray?logo=google&amp"></a> -->
    <!-- <a href="https://colab.research.google.com/drive/1bB3LqAzceNTW2urXeMpOTPoJYTRoKlzB?usp=sharing"><img src="https://img.shields.io/badge/Colab-gray?logo=Google%20Colab&amp"></a> -->
    <!-- <a href="https://huggingface.co/spaces/H-Liu1997/EMAGE"><img src="https://img.shields.io/badge/Huggingface Space-gray?logo=huggingface&amp"></a> -->
    <a href="https://www.youtube.com/watch?v=T0OYPvViFGE"><img src="https://img.shields.io/badge/ACMMM 2022-gray?logo=arxiv&amp"></a>
</div>

<h2>1. News</h2>

Welcome volunteers to contribute and collaborate on related topics. Feel free to submit the pull requests! Currently this repo is mainly maintained by haiyangliu1997@gmail.com in freetime.
 
- **[2025/01]** New inference api, visualization api, evaluation api, training codebase, are available!
- **[2024/07]** Now you could [download smplx motion (in .npz) file](https://huggingface.co/spaces/H-Liu1997/EMAGE), visualize with our blender addon and retarget to your avatar!
- **[2024/04]** Thanks to [@camenduru](https://twitter.com/camenduru), Replicate version EMAGE is available! you can directly call EMAGE via API!
- **[2024/03]** Thanks to [@sunday9999](https://github.com/sunday9999) for speeding up the inference video rendering from 1000s to 25s! 
- **[2024/03]** EMAGE Demos: [Colab demo](https://colab.research.google.com/drive/1AINafluW6Ba5-KYN_L43eyFK0zRklMvr?usp=sharing), [Gradio demo](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024#user-content--gradio-demo).
- **[2024/02]** Quick Access: [How to setup EMAGE](https://github.com/PantoMatrix/PantoMatrix/blob/main/scripts/EMAGE_2024/readme.md), [Details of BEAT](https://github.com/PantoMatrix/PantoMatrix/blob/main/scripts/BEAT_2022/readme_beat.md). 🚀!
- **[2024/02]** Thanks to [@wubowen416](https://github.com/wubowen416) for the [scripts of automatic video visualization #83](https://github.com/PantoMatrix/PantoMatrix/issues/83) during inference!
- **[2022/03]** CaMN training scripts from [anonymous submission](https://github.com/beat2022dataset/beat).
<!-- - **[2024/02]** Training and Inference [Scripts](https://github.com/PantoMatrix/PantoMatrix/blob/main/scripts/EMAGE_2024/readme.md) are available for [EMAGE](https://pantomatrix.github.io/EMAGE/). -->
<!-- - **[2023/12]** [EMAGE](https://pantomatrix.github.io/EMAGE/) is available, including BEATX with both FLAME head and SMPLX body parameters.
- **[2023/05]** [BEAT_GENEA](https://drive.google.com/file/d/1wYW7eWAYPkYZ7WPOrZ9Z_GIll13-FZfx/view?usp=share_link) is allowed for pretraining in [GENEA2023](https://genea-workshop.github.io/2023/challenge/)! Thanks for GENEA's organizers! 
- **[2023/03]** [Samples](https://drive.google.com/drive/folders/1YLoGaJcrhp9Ap2tsJ4A5xNbKpzmDX6yD?usp=share_link) and [readme](https://github.com/PantoMatrix/BEAT/tree/main/beat2smpl) for SMPL-X body and hands data.
- **[2023/01]** [English data v0.2.1](https://drive.google.com/file/d/1Akf0WgAwuH2fvlWbvNpif4XRqXlpznh9/view?usp=share_link) are available. Fix the orientation issue. See [updation](./docs/updation.md) for details.
- **[2023/01]** Provide checkpoints (#14, #16), scripts for rendering (#17), preprocessing (#18).  
- **[2022/12]** Provide English data in Zip files (#10).
- **[2022/10]** [Project page](https://pantomatrix.github.io/BEAT/) and [rendered videos](https://drive.google.com/drive/folders/1ghZ7_4LkCyM_IZxTElzAwPzGheLrBGBu) are available.
- **[2022/08]** [All languages data v0.1.0](https://drive.google.com/drive/folders/1CVyJOp3G_A9l1N_CsKdHgXQfB4pXhG8c?usp=share_link)  (in separated files) are available. -->


 <!-- fgd: 2.2332441801
  bc: 0.7650137509
  l1: 9.8172253088 -->

<br>

## 2. Download links

We summarize the download links for pretrained models, datasets, tools supported as below.

### Model list

| Model  | Paper           | Inputs | Outputs**         | Language (Train)         | Performance*** (Full Body FGD) | Weights |
|--------|-----------------|--------|------------------|--------------------------|-------------------------------|---------|
| DisCo  | ACMMM 2022     | Audio  | Upper + Hands    | English (Speaker 2) | 2.233                            |    [Link](https://huggingface.co/H-Liu1997/disco_audio/tree/main)     |
| CaMN   | ECCV 2022      | Audio  | Upper + Hands    | English (Speaker 2) | 2.120                         |     [Link](https://huggingface.co/H-Liu1997/camn_audio/tree/main)    |
| EMAGE  | CVPR 2024      | Audio  | Full Body + Face | English (Speaker 2) | 0.615                        |    [Link](https://huggingface.co/H-Liu1997/emage_audio/tree/main)     |

** Outputs are in SMPLX and FLAME parameters. 

<!-- *** Full performance report including more metrics is [here](#performance-report). -->


## Dataset and tools


| Type   |  |  |         |
|--------|----------------------|--------------------|---------|
| Datasets  | [BEAT2 (SMPLX+FLAME)](https://huggingface.co/datasets/H-Liu1997/BEAT2) | [BEAT (BVH + ARKit)]()    | FGD Eval |
| Tools  | [Blender Addon](https://huggingface.co/datasets/H-Liu1997/BEAT2_Tools/resolve/main/smplx_blender_addon_20230921.zip?download=true) | SMPLX-FLAME Model | ARKit2FLAME |



<br>


## 3. How to Use the API (Inference)

### 3.1. Generate Gestures from Speech

You will generate SMPLX and FLAME parameters in this step and store into `.npz` files

#### Approach 1: Using Hugging Face Space
Upload your audio and directly download the results from our Hugging Face Space.

#### Approach 2: Local Setup
Clone the repository and set up locally.

```bash
# Clone the repository
git clone <repo_url>
bash setup.sh

# Run the test script
python test.py --audio_dir /your_audio_dir --save_dir /your_save_dir --visualization
```

#### Approach 3: Call API Directly

```python
# copy the ./models folder iin your project folder
from .model.camn_audio import CaMNAudioModel

model = CaMNAudioModel.from_pretrained("H-Liu1997/huggingface-model/camn_audio")
model.cuda().eval()

import librosa
import numpy as np
import torch
# copy the ./emage_utils folder in your project folder
from emage_utils import beat_format_save

audio_np, sr = librosa.load("/audio_path.wav", sr=model.cfg.audio_sr)
audio = torch.from_numpy(audio_np).float().cuda().unsqueeze(0)

motion_pred = model(audio)["motion_axis_angle"]
motion_pred_np = motion_pred.cpu().numpy()
beat_format_save(motion_pred_np, "/result_motion.npz")
```

<br>


### 3.2. Visualization

When you run the scripts in 3.1. there is an parameter `--visualization` to automatic enable visualizaion. Besides, you could also try visualiztion by the below.

#### Approach 1: Blender (Recommended)
Render the output using Blender or our visualization function:



#### Approach 2: 3D mesh

```python
# render a npz file to a mesh video
from emage_utils import fast_render
fast_render.render_one_sequence_no_gt("/result_motion.npz", "/audio_path.wav", "/result_video.mp4", remove_global=True)
```

#### Approach 3: 2D OpenPose style video

```python
from trochvision.io import write_video
from emage_utils.format_transfer import render2d
from emage_utils import fast_render


motion_dict = np.load(npz_path, allow_pickle=True)
# face
v2d_face = render2d(motion_dict, (512, 512), face_only=True, remove_global=True)
write_video(npz_path.replace(".npz", "_2dface.mp4"), v2d_face.permute(0, 2, 3, 1), fps=30)
fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dface.mp4"), audio_path, npz_path.replace(".npz", "_2dface_audio.mp4"))

# body
v2d_body = render2d(motion_dict, (720, 480), face_only=False, remove_global=True)
write_video(npz_path.replace(".npz", "_2dbody.mp4"), v2d_body.permute(0, 2, 3, 1), fps=30)
fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dbody.mp4"), audio_path, npz_path.replace(".npz", "_2dbody_audio.mp4"))
```

<br>

###  3.3. Evaluation

For academic users, the evaluation code is organized into an evaluation API. 

```python
# copy the ./emage_evaltools folder into your folder
from emage_evaltools.metric import FGD, BC, L1Div, LVDFace, MSEFace

# init
fgd_evaluator = FGD(download_path="./emage_evaltools/")
bc_evaluator = BC(download_path="./emage_evaltools/", sigma=0.3, order=7)
l1div_evaluator= L1div()
lvd_evaluator = LVDFace()
mse_evaluator = MSEFace()

# Example usage
for motion_pred in all_motion_pred:
    # bc and l1 require position representation
    motion_position_pred = get_motion_rep_numpy(motion_pred, device=device, betas=betas)["position"] # t*55*3
    motion_position_pred = motion_position_pred.reshape(t, -1)
    # ignore the start and end 2s, this may for beat dataset only
    audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=2 * 16000, t_end=int((t-60)/30*16000))
    motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=60, t_end=t-60, pose_fps=30, without_file=True)
    bc_evaluator.compute(audio_beat, motion_beat, length=t-120, pose_fps=30)

    l1_evaluator.compute(motion_position_pred)
    
    face_position_pred = get_motion_rep_numpy(motion_pred, device=device, expressions=expressions_pred, expression_only=True, betas=betas)["vertices"] # t -1
    face_position_gt = get_motion_rep_numpy(motion_gt, device=device, expressions=expressions_gt, expression_only=True, betas=betas)["vertices"]
    lvd_evaluator.compute(face_position_pred, face_position_gt)
    mse_evaluator.compute(face_position_pred, face_position_gt)
    
    # fgd requires rotation 6d representaiton
    motion_gt = torch.from_numpy(motion_gt).to(device).unsqueeze(0)
    motion_pred = torch.from_numpy(motion_pred).to(device).unsqueeze(0)
    motion_gt = rc.axis_angle_to_rotation_6d(motion_gt.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
    motion_pred = rc.axis_angle_to_rotation_6d(motion_pred.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
    fgd_evaluator.update(motion_pred.float(), motion_gt.float())
    
metrics = {}
metrics["fgd"] = fgd_evaluator.compute()
metrics["bc"] = bc_evaluator.avg()
metrics["l1"] = l1_evaluator.avg()
metrics["lvd"] = lvd_evaluator.avg()
metrics["mse"] = mse_evaluator.avg()
```

Hyperparameters may vary depending on the dataset. For example, for the BEAT dataset, we use `(0.3, 7)`; for the TalkShow dataset, we use `(0.5, 7)`. You may adjust based on your data.

<br>

## 4. Training

This new codebase only have the audio-only version model for better real-world applications. For reproducing audio+text results in the paper, please check and reference the previous codebase below.

| Model  | Inputs (Paper)      | Old Codebase | Input (Current Codebase)  | 
|--------|---------------------|-----------------------------|---------------------|
| DisCo  | Audio + Text        |    link                         | Audio               |       
| CaMN   | Audio + Text + Emotion + Facial |      link               | Audio               |          
| EMAGE  | Audio + Text        | link                      | Audio               |     


### 4.1. General Setup

Environment setup, skip if you already setup the inference.
```bash
bash setup.sh
source /content/py39/bin/activate
```

Download the dataset
```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/H-Liu1997/BEAT2
```

Your folder should like follows for the correct path
```bash
/content/
|-- beat2
|-- pantomatrix-master
   `-- train_emage_audio.py
```

### 4.1. Training EMAGE

#### Preprocessing

Extract the foot contact data
```bash
cd ./pantomatrix-master/
python ./datasets/foot_contact.py
```

#### VQ-VAEs

Use pretrained models:

```python
from .model.emage_audio import EmageVQVAEModel, EmageVAEModel, EmageVQModel

motion_vq = EmageVQModel(
  face_model=EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/face"),
  upper_model=EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/upper"),
  lower_model=EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/lower"),
  hands_model=EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/hands"),
  global_model=EmageVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/global"),
).to(device)
motion_vq.eval()
```

#### Audio2Gesture Network

```bash
torchrun --nproc_per_node 1 --nnodes 1 train_emage_audio.py --config ./configs/emage_audio.py --evaluation
```

Use these flags as needed:

- `--evaluation`: Calculate the test metric.
- `--wandb`: Activate logging to WandB.
- `--visualization`: Render test results (slow; disable for efficiency).
- `--test`: Test mode; load last checkpoint and evaluate.
- `--debug`: Debug mode; iterate one data point for fast testing.

<br>

### 4.2. Training CaMN

#### Audio2Gesture Network

```bash
torchrun --nproc_per_node 1 --nnodes 1 train_camn_audio.py --config ./configs/camn_audio.py --evaluation
```

Use these flags as needed:

- `--evaluation`: Calculate the test metric.
- `--wandb`: Activate logging to WandB.
- `--visualization`: Render test results (slow; disable for efficiency).
- `--test`: Test mode; load last checkpoint and evaluate.
- `--debug`: Debug mode; iterate one data point for fast testing.

### 4.3. Training DisCo

#### Preprocessing

Extract the cluster information
```bash
cd ./pantomatrix-master/
python ./datasets/clustering.py
```

#### Audio2Gesture Network

```bash
torchrun --nproc_per_node 1 --nnodes 1 train_disco_audio.py --config ./configs/disco_audio.py --evaluation
```

Use these flags as needed:

- `--evaluation`: Calculate the test metric.
- `--wandb`: Activate logging to WandB.
- `--visualization`: Render test results (slow; disable for efficiency).
- `--test`: Test mode; load last checkpoint and evaluate.
- `--debug`: Debug mode; iterate one data point for fast testing.

<br>

<h2>5. Reference </h2>

<!-- **CoRR 2024**<br /> -->
**CVPR 2024**<br />
**EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling**<br >
<sub>
<a href="https://h-liu1997.github.io/">Haiyang Liu</a>\*,
<a href="https://zzhat0706.github.io/PersonalPage/">Zihao Zhu</a>\*,
<a href="https://ps.is.mpg.de/person/gbecherini">Giorgio Becherini</a>, 
<a href="https://scholar.google.com/citations?user=9sWVrREAAAAJ&hl=en">Yichen Peng</a>,
<a>Mingyang Su</a>,
<a>You Zhou</a>,
<a href="https://iwanao731.github.io/">Naoya Iwamoto</a>,
<a href="http://www.bozheng-lab.com/">Bo Zheng</a>,
<a href="https://ps.is.mpg.de/person/black">Michael J. Black</a>\
</sub>
<sub>
<sup>(*Equal Contribution)</sup>
<sub>
------------
<img src ="assets/EMAGE_2024/teaser.gif" width="100%">

<p align="left">
We propose EMAGE, a framework to generate full-body human gestures from audio and masked gestures, encompassing facial, local body, hands, and global movements. To achieve this, we first introduce BEATX (BEAT-SMPLXFLAME), a new mesh-level holistic co-speech dataset. BEATX combines MoShed SMPLX body with FLAME head parameters and further refines the modeling of head, neck, and finger movements, offering a community-standardized, high-quality 3D motion captured dataset. EMAGE leverages masked body gesture priors during training to boost inference performance. It involves a Masked Audio Gesture Transformer, facilitating joint training on audio-togesture generation and masked gesture reconstruction to effectively encode audio and body gesture hints. Encoded body hints from masked gestures are then separately employed to generate facial and body movements. Moreover, EMAGE adaptively merges speech features from the audio’s rhythm and content and utilizes four compositional VQVAEs to enhance the results’ fidelity and diversity. Experiments demonstrate that EMAGE generates holistic gestures with state-of-the-art performance and is flexible in accepting predefined spatial-temporal gesture inputs, generating complete, audio-synchronized results.
</p>

<p align="center">
<img src ="assets/EMAGE_2024/res.png" width="100%">
</p>

<p align="center">
-
<a href="https://pantomatrix.github.io/EMAGE/">Project Page</a>
-
<a href="https://arxiv.org/abs/2401.00374">Paper</a>
-
<a href="https://www.youtube.com/watch?v=T0OYPvViFGE">Video</a>
-
<a href="https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024">Code</a>
-
<a href="https://colab.research.google.com/drive/1AINafluW6Ba5-KYN_L43eyFK0zRklMvr?usp=sharing">Demo</a>
-
<a href="https://drive.google.com/drive/folders/1ukbifhHc85qWTzspEgvAxCXwn9mK4ifr">Dataset</a>
-
<a href="https://drive.google.com/drive/folders/1ukbifhHc85qWTzspEgvAxCXwn9mK4ifr">Blender Add-On</a>
-
</p>

<p align=center>
    <a href="https://www.youtube.com/watch?v=T0OYPvViFGE">
    <img  width="68%" src="assets/EMAGE_2024/th.png">
    </a>
</p>

<!-- <p align="center">
-
<a>Data Processing</a>
-
</p>
<img src ="assets/EMAGE_2024/data.png" width="100%"> -->

------------

**ECCV 2022**<br />
**BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis**<br >
<sub>
<a href="https://h-liu1997.github.io/">Haiyang Liu</a>,
<a href="https://zzhat0706.github.io/PersonalPage/">Zihao Zhu</a>,
<a href="https://iwanao731.github.io/">Naoya Iwamoto</a>,
<a href="https://scholar.google.com/citations?user=9sWVrREAAAAJ&hl=en">Yichen Peng</a>,
<a href="https://scholar.google.co.jp/citations?user=hgCoNowAAAAJ&hl=ja">Zhengqing Li</a>,
<a>You Zhou</a>,
<a href="https://scholar.google.com.sg/citations?user=Bm1TcmsAAAAJ&hl=en">Elif Bozkurt</a>,
<a href="http://www.bozheng-lab.com/">Bo Zheng</a>
</sub>
------------
<img src ="assets/BEAT_2022/teaser.png" width="100%">

<p align="left">
Achieving realistic, vivid, and human-like synthesized conversational gestures conditioned on multi-modal data is still an unsolved problem due to the lack of available datasets, models and standard evaluation metrics. To address this, we build Body-Expression-Audio-Text dataset, BEAT, which has i) 76 hours, high-quality, multi-modal data captured from 30 speakers talking with eight different emotions and in four different languages, ii) 32 millions frame-level emotion and semantic relevance annotations. Our statistical analysis on BEAT demonstrates the correlation of conversational gestures with facial expressions, emotions, and semantics, in addition to the known correlation with audio, text, and speaker identity. Based on this observation, we propose a baseline model, Cascaded Motion Network (CaMN), which consists of above six modalities modeled in a cascaded architecture for gesture synthesis. To evaluate the semantic relevancy, we introduce a metric, Semantic Relevance Gesture Recall (SRGR). Qualitative and quantitative experiments demonstrate metrics’ validness, ground truth data quality, and baseline’s state-of-the-art performance. To the best of our knowledge, BEAT is the largest motion capture dataset for investigating human gestures, which may contribute to a number of different research fields, including controllable gesture synthesis, cross-modality analysis, and emotional gesture recognition.
</p>

<p align="center">
<img src ="assets/BEAT_2022/data2.png" width="100%">
</p>

<p align="center">
-
<a href="https://pantomatrix.github.io/BEAT/">Project Page</a>
-
<a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670605.pdf">Paper</a>
-
<a href="https://www.youtube.com/watch?v=F6nXVTUY0KQ">Video</a>
-
<a href="https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/BEAT_2022">Code</a>
-
<a href="https://colab.research.google.com/drive/1bB3LqAzceNTW2urXeMpOTPoJYTRoKlzB?usp=sharing">Colab Demo</a>
-
<a href="https://pantomatrix.github.io/BEAT-Dataset/">Dataset</a>
-
<a href="https://paperswithcode.com/sota/gesture-generation-on-beat?p=beat-a-large-scale-semantic-and-emotional">Benchmark</a>
-
</p>

<p align=center>
    <a href="https://www.youtube.com/watch?v=F6nXVTUY0KQ">
    <img  width="68%" src="assets/BEAT_2022/th.png">
    </a>
</p>

<!-- <p align="center">
-
<a>Data Distribution</a>
-
</p>
<img src ="assets/BEAT_2022/data1.png" width="100%">
<img src ="assets/BEAT_2022/data2.png" width="100%"> -->

------------

**ACMMM 2022**<br />
**DisCo: Disentangled Implicit Content and Rhythm Learning for Diverse Co-Speech Gesture Synthesis**<br >
<sub>
<a href="https://h-liu1997.github.io/">Haiyang Liu</a>,
<a href="https://iwanao731.github.io/">Naoya Iwamoto</a>,
<a href="https://zzhat0706.github.io/PersonalPage/">Zihao Zhu</a>,
<a href="https://scholar.google.co.jp/citations?user=hgCoNowAAAAJ&hl=ja">Zhengqing Li</a>,
<a>You Zhou</a>,
<a href="https://scholar.google.com.sg/citations?user=Bm1TcmsAAAAJ&hl=en">Elif Bozkurt</a>,
<a href="http://www.bozheng-lab.com/">Bo Zheng</a>
</sub>
------------
<img src ="assets/DisCo_2022/teaser.png" width="100%">

<p align="left">
Current co-speech gestures synthesis methods struggle with generating diverse motions and typically collapse to single or few frequent motion sequences, which are trained on original data distribution with customized models and strategies. We tackle this problem by temporally clustering motion sequences into content and rhythm segments and then training on content-balanced data distribution. In particular, by clustering motion sequences, we have observed for each rhythm pattern, some motions appear frequently, while others appear less. This imbalance results in the difficulty of generating low frequent occurrence motions and it cannot be easily solved by resampling, due to the inherent many-tomany mapping between content and rhythm. Therefore, we present DisCo, which disentangles motion into implicit content and rhythm features by contrastive loss for adopting different data balance strategies. Besides, to model the inherent mapping between content and rhythm features, we design a diversity-and-inclusion network (DIN), which firstly generates content features candidates and then selects one candidate by learned voting. Experiments on two public datasets, Trinity and S2G-Ellen, justify that DisCo generates more realistic and diverse motions than state-of-the-art methods.
</p>

<p align="center">
<img src ="assets/DisCo_2022/res.png" width="100%">
</p>

<p align="center">
-
<a href="https://pantomatrix.github.io/DisCo/">Project Page</a>
-
<a href="https://dl.acm.org/doi/abs/10.1145/3503161.3548400">Paper</a>
-
<a href="https://www.youtube.com/watch?v=Nd6NX27ykgA">Video</a>
-
<a href="https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/DisCo_2022">Code</a>
-
</p>

<p align=center>
    <a href="https://www.youtube.com/watch?v=Nd6NX27ykgA">
    <img  width="68%" src="assets/DisCo_2022/th.png">
    </a>
</p>



Copyright Information
============
The website is inspired by the template of <a href="https://github.com/sebastianstarke/AI4Animation">AI4Animation</a>.
