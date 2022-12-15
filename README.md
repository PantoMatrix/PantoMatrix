# BEAT(CaMN) — Official PyTorch implementation 
## New
- [Project Page](https://pantomatrix.github.io/BEAT/). [Download Dataset](https://drive.google.com/drive/folders/1CVyJOp3G_A9l1N_CsKdHgXQfB4pXhG8c). [Rendered Results](https://drive.google.com/drive/folders/1ghZ7_4LkCyM_IZxTElzAwPzGheLrBGBu)
- This repository is as same as our anonymous [submission in 202203](https://github.com/beat2022dataset/beat)


## Contents
- train and inference scripts
    - CaMN (ours)
    - End2End (ours)
    - Motion AutoEncoder (for evaluation)
    - data preprocessing
        - load specific number of joints with predefined FPS from bvh
        - build word2vec model 
        - cache generation (.lmdb) 
- dataset examples in [beat.zip](https://drive.google.com/file/d/1bHiOi7UQwoCZ3sMuzDtBJII8nLVE-csa/view?usp=sharing)
    - original files to generate cache in train/val/test
    - cache for language_model, pretrained_vae

## Train
0. `python == 3.7`
1. build folders like:
    - codes
    - datasets
    - outputs
2. download the scripts to `codes/beat/`    
3. extract beat.zip to datasets/beat
4. run ```pip install -r requirements.txt``` in the path `./codes/beat/` 
5. run ```python train.py -c ./configs/camn.yaml``` for training and inference.
6. load ```./outputs/exp_name/119/res_000_008.bvh``` into blender to visualize the test results.

## Modifiaction

- train End2End model, add `g_name: PoseGenerator` in camn.yaml
- generate data cache from stratch
    - `cd ./dataloaders && python bvh2anyjoints.py` for motion data
    - `cd ./dataloaders && python build_vocab.py` for language model
- remove modalities, e.g., remove facial expressions.
    - set `facial_rep: None` and `facial_f: 0` in camn.yaml
    - ``` python train.py -c ./configs/camn.yaml ```
    - for semantic-weighted loss, set `sem_weighted == False` in camn_trainer.py
- refer to `./utils/config.py` for other parameters.
