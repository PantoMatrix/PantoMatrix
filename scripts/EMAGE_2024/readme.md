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
# Prerequisite for rendering
conda install -c conda-forge libstdcxx-ng
# Install with pip:
pip install -r ./scripts/EMAGE_2024/requirements.txt
```

## Download weights
You may run the following command to download weights:

```shell
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
`-- emage_240.bin
```

# 🚀 Training and Inference 

## Inference

Here is the cli command for running inference scripts:

```shell
python scripts/EMAGE_2024/test_demo.py --config ./scripts/EMAGE_2024/configs/emage_test.yaml
```

## <span id="train"> Training </span>

### Data Preparation

Download the unzip version BEATX via hugging face: 

```shell
git lfs install
git clone https://huggingface.co/datasets/H-Liu1997/BEATX
```

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

- Ongoing
