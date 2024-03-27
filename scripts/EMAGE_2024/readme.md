# 📝 Release Plans

- [x] Inference codes and pretrained weights
- [x] Training scripts

# ⚒️ Installation

## Build Environtment

We Recommend a python version `==3.8` and cuda version `==12.2`. Then build environment as follows:

```shell
# [Optional] Create a virtual env
git clone https://github.com/PantoMatrix/PantoMatrix.git
conda create -n emagepy38 python==3.8
conda activate emagepy38
# Install ffmpeg for media processing and libstdcxx-ng for rendering
conda install -c conda-forge libstdcxx-ng ffmpeg
# Install with pip:
pip install -r ./scripts/EMAGE_2024/requirements.txt
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

## Download weights
You may run the following command to download weights in ```<your root>/PantoMatrix/```:

```shell
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/H-Liu1997/EMAGE
```
These weights should be orgnized as follows:

```text
<your root>/PantoMatrix/EMAGE/
|-- pretrained_vq
|   |-- hands_vertex_1layer_710.bin
|   |-- last_790_face_v2.bin
|   |-- last_1700_foot.bin
|   |-- lower_foot_600.bin
|   `-- upper_vertex_1layer_710.bin
|-- smplx_models
|   `-- smplx/SMPLX_NEUTRAL_2020.npz
|-- test_sequences
|-- emage_audio_175.bin
`-- emage_240.bin
```

# 🚀 Training and Inference 

## Inference

Here is the command for running inference scripts under the path ```<your root>/PantoMatrix/```, it will take around 1 min to generate motion. You can visualize the result .npz files via blender using our blender addon. If you want to visualize with video, you may try our gradio demo.

```shell
python scripts/EMAGE_2024/test_demo.py --config ./scripts/EMAGE_2024/configs/emage_test.yaml
```

## <span id="train"> Training </span>

### Data Preparation

Download the unzip version BEAT2 via hugging face in path ```<your root>/PantoMatrix/```: 

```shell
git lfs install
git clone https://huggingface.co/datasets/H-Liu1997/BEAT2
```

### Evaluation of Pretrianed Weights

Once you downloaded full BEAT2 dataset, run:
```shell
python scripts/EMAGE_2024/test.py --config ./scripts/EMAGE_2024/configs/emage.yaml
```

<!-- You may get the results:
```
 01-15 10:42:27 | l2 loss: 7.630602982715039e-08
 01-15 10:42:27 | lvel loss: 7.505888918822572e-05
 01-15 10:42:27 | fid score: 0.5514388420395155
 01-15 10:42:27 | align score: 0.772429069711832
 01-15 10:42:27 | l1div score: 13.0666241037962
 01-15 10:42:27 | total inference time: 40 s for 956 s motion
``` -->


### Training of EMAGE

```shell
python scripts/EMAGE_2024/train.py --config ./scripts/EMAGE_2024/configs/emage.yaml 
```

### Training of VQVAE

```shell
python scripts/EMAGE_2024/train.py --config ./scripts/EMAGE_2024/configs/cnn_vqvae_face_30.yaml 
```
```shell
python scripts/EMAGE_2024/train.py --config ./scripts/EMAGE_2024/configs/cnn_vqvae_hands_30.yaml 
```
```shell
python scripts/EMAGE_2024/train.py --config ./scripts/EMAGE_2024/configs/cnn_vqvae_lower_30.yaml 
```
```shell
python scripts/EMAGE_2024/train.py --config ./scripts/EMAGE_2024/configs/cnn_vqvae_lower_foot_30.yaml 
```
```shell
python scripts/EMAGE_2024/train.py --config ./scripts/EMAGE_2024/configs/cnn_vqvae_upper_30.yaml 
```

### Other baselines, e.g., CaMN

```shell
python scripts/EMAGE_2024/train.py --config ./scripts/EMAGE_2024/configs/camn.yaml 
```

# 🎨 Gradio Demo

###  Finish the [environment setup](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024#build-environtment) & [weight download](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024#download-weights) steps.

### Run the demo, then open the localhost link in browser.

```shell
python scripts/EMAGE_2024/demo.py
```
### Upload audio (less than 60s), then hit the submit, and output result will be rendered in output for 3 minutes.

<img src ="https://github.com/PantoMatrix/PantoMatrix/blob/main/assets/EMAGE_2024/afrun.jpg" width="100%">