<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/PantoMatrix/BEAT/master/docs/assets/teaser2.png" />
</p>

<h1 style="text-align: center;">
BEAT: Body-Expression-Audio-Text Dataset 
</h1>
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beat-a-large-scale-semantic-and-emotional/gesture-generation-on-beat)](https://paperswithcode.com/sota/gesture-generation-on-beat?p=beat-a-large-scale-semantic-and-emotional)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/H-Liu1997/BEAT/tree/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bB3LqAzceNTW2urXeMpOTPoJYTRoKlzB?usp=sharing)
 
# News 
- **[2023/12]** Our new work [EMAGE](https://pantomatrix.github.io/EMAGE/) is available, including BEATX with both FLAME head and SMPLX body parameters. The new [data](https://drive.google.com/drive/folders/1ukbifhHc85qWTzspEgvAxCXwn9mK4ifr?usp=drive_link) and [codes](https://github.com/PantoMatrix/BEAT/tree/main/scripts/EMAGE_2024) are available.
- **[2023/05]** [BEAT_GENEA](https://drive.google.com/file/d/1wYW7eWAYPkYZ7WPOrZ9Z_GIll13-FZfx/view?usp=share_link) is allowed for pretraining in [GENEA2023](https://genea-workshop.github.io/2023/challenge/)! Thanks for GENEA's organizers! 
- **[2023/03]** [Samples](https://drive.google.com/drive/folders/1YLoGaJcrhp9Ap2tsJ4A5xNbKpzmDX6yD?usp=share_link) and [readme](https://github.com/PantoMatrix/BEAT/tree/main/beat2smpl) for SMPL-X body and hands data.
- **[2023/01]** [English data v0.2.1](https://drive.google.com/file/d/1Akf0WgAwuH2fvlWbvNpif4XRqXlpznh9/view?usp=share_link) are available. Fix the orientation issue. See [updation](./docs/updation.md) for details.
- **[2023/01]** Provide checkpoints (#14, #16), scripts for rendering (#17), preprocessing (#18).  
- **[2022/12]** Provide English data in Zip files (#10).
- **[2022/10]** [Project page](https://pantomatrix.github.io/BEAT/) and [rendered videos](https://drive.google.com/drive/folders/1ghZ7_4LkCyM_IZxTElzAwPzGheLrBGBu) are available.
- **[2022/08]** [All languages data v0.1.0](https://drive.google.com/drive/folders/1CVyJOp3G_A9l1N_CsKdHgXQfB4pXhG8c?usp=share_link)  (in separated files) are available.
- **[2022/03]** CaMN training scripts from [anonymous submission](https://github.com/beat2022dataset/beat).


# Features
- **10-Scale Semantic Relevancy:** BEAT provides a score and category-label for semantic relevancy between gestures and speech content: no gestures (0), beat gestures (1), low-middle-high quaility deictic gestures (2,3,4), iconic gestures (5,6,7), metaphoric gestures (8,9,10). 
- **8-Class Emotional Gestures:** For each speaker, data in speech section are recorded with eight emotions: neutral, happiness, anger, sadness, contempt, surprise, fear, and disgust. Data in conversation are labeled as neutral.
- **4-Modality Captured Data:** With 16 cameras motion capture system and iphone arkit, BEAT recorded data in four modalities: 75 joints' motion, 52 dimensions blendshape weights, audio and text. 
- **76-Hour and 30-Speaker:** BEAT (English data) consists of 10 recorded four hours speakers and 20 recorded one hour speakers.
- **4-Language:** BEAT contains four types of languages: English (60h), Chinese (12h), Spanish (2h) and Japanese (2h). For the latter three languages, speakers also record English data to provide paired data.
- **2-Scenario:** BEAT provides speech (50%) and conversation (50%) recording. 

# Benchmark
### Gesture Generation on BEAT-16h (speaker 2,4,6,8 in [English data v0.2.1](https://drive.google.com/file/d/1Akf0WgAwuH2fvlWbvNpif4XRqXlpznh9/view?usp=share_link))
| Method             | Venue             | Input Modalities    | FID**         | SRGR          | BeatAlign      | ckpt |
|--------------------|-------------------|---------------------|---------------|---------------|----------------|------|
| [Seq2Seq](https://arxiv.org/abs/1810.12541)            | ICRA'19           | text                | 261.3         | 0.173         | 0.729          | -        |
| [Speech2Gestures](https://arxiv.org/abs/1906.04160)    | CVPR'19           | audio               | 256.7         | 0.092         | 0.751          | -        |
| [Audio2Gestures](https://ieeexplore.ieee.org/document/9710107)     | ICCV'21           | audio               | 223.8         | 0.097         | 0.766          | -        |
| [MultiContext](https://arxiv.org/abs/2009.02119)       | SIGGRAPH ASIA'20  | audio, text         | 176.2 (177.2*)| 0.195 (0.227) | 0.776 (0.751)  | [link](https://drive.google.com/file/d/1j3zNf1FyAL4qR4Y5JhqVlULBghKSlCAi/view?usp=share_link) |
| [CaMN](https://arxiv.org/abs/2203.05297)               | ECCV'22           | audio, text, facial | 123.7 (122.8) | 0.239 (0.240) | 0.783 (0.782)  | [link](https://drive.google.com/file/d/1Q7v_e3K_cgR93a70hlen7KAf2K94NU7j/view?usp=share_link) |

*Checkpoints results trained from this repo. are denoted in parentheses. Results in paper are from codes: [seq2seq](https://github.com/youngwoo-yoon/Co-Speech_Gesture_Generation), [s2g](https://github.com/amirbar/speech2gesture), [a2g](https://github.com/JingLi513/Audio2Gestures), [mutli](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context), [camn](https://github.com/beat2022dataset/beat). 

**Pretrained [300D AutoEncoder](https://drive.google.com/file/d/1kgUramYf8US2pg1lssJBflDqV3_CJpqH/view?usp=share_link) for FID calculation.



# Dataset 
### Introcution
- name: `1_wayne_0_1_8` 
   - where `1_wayne` is speaker id and name, 
   - `0` is the recording type: `0` English Speech, `1` English Conversation, `2` Chinese Speech, `3` Chinese Conversation, `4` Spanish Speech, `5` Spanish Conversation, `6` Japanese Speech, `7` Japanese Conversation.
   - `1_8` is the `start` and `end` id for the current sequence, where range is `1-118` for speech and `1-12` for conversation.
   - for speech section: `0-64` neutral, `65-72` happiness, `73-80` anger, `81-86` sadness, `87-94` contempt, `95-102` surprise, `103-110` fear, `111-118` disgust.
- format:
   - `120` FPS `.bvh` for motion, using `Z-up`, `Y-forward` in blender, right-hand system.
   - `60` FPS `.json` for facial blendshape weights.
   - `16K` HZ `.wav` for audio.
   - `.TextGrid` for text
   - `.csv` for emotion label, `0-7`: neutral, happiness, anger, sadness, contempt, surprise, fear, and disgust.  
   - `.txt` for semantic label, in `types`, `start`, `end`, `duration`, `score`, and `keywords`. 
- missing sequences:
   - speaker 9: `0_2_8`, speaker 21: `0_1_8`.       

### Train/val/test split
Script is in `/dataloaders/preprocessing.ipynb`, ratio: `2880:500:500`
```python
split_rule_english = {
    # 4h speakers x 10
    "1, 2, 3, 4, 6, 7, 8, 9, 11, 21":{
        # 48+40+100=188mins each
        "train": [
            "0_9_9", "0_10_10", "0_11_11", "0_12_12", "0_13_13", "0_14_14", "0_15_15", "0_16_16", \
            "0_17_17", "0_18_18", "0_19_19", "0_20_20", "0_21_21", "0_22_22", "0_23_23", "0_24_24", \
            "0_25_25", "0_26_26", "0_27_27", "0_28_28", "0_29_29", "0_30_30", "0_31_31", "0_32_32", \
            "0_33_33", "0_34_34", "0_35_35", "0_36_36", "0_37_37", "0_38_38", "0_39_39", "0_40_40", \
            "0_41_41", "0_42_42", "0_43_43", "0_44_44", "0_45_45", "0_46_46", "0_47_47", "0_48_48", \
            "0_49_49", "0_50_50", "0_51_51", "0_52_52", "0_53_53", "0_54_54", "0_55_55", "0_56_56", \
            
            "0_66_66", "0_67_67", "0_68_68", "0_69_69", "0_70_70", "0_71_71",  \
            "0_74_74", "0_75_75", "0_76_76", "0_77_77", "0_78_78", "0_79_79",  \
            "0_82_82", "0_83_83", "0_84_84", "0_85_85",  \
            "0_88_88", "0_89_89", "0_90_90", "0_91_91", "0_92_92", "0_93_93",  \
            "0_96_96", "0_97_97", "0_98_98", "0_99_99", "0_100_100", "0_101_101",  \
            "0_104_104", "0_105_105", "0_106_106", "0_107_107", "0_108_108", "0_109_109",  \
            "0_112_112", "0_113_113", "0_114_114", "0_115_115", "0_116_116", "0_117_117",  \
            
            "1_2_2", "1_3_3", "1_4_4", "1_5_5", "1_6_6", "1_7_7", "1_8_8", "1_9_9", "1_10_10", "1_11_11",
        ],
        # 8+7+10=25mins each
        "val": [
            "0_57_57", "0_58_58", "0_59_59", "0_60_60", "0_61_61", "0_62_62", "0_63_63", "0_64_64", \
            "0_72_72", "0_80_80", "0_86_86", "0_94_94", "0_102_102", "0_110_110", "0_118_118", \
            "1_12_12",
        ],
        # 8+7+10=25mins each
        "test": [
           "0_1_1", "0_2_2", "0_3_3", "0_4_4", "0_5_5", "0_6_6", "0_7_7", "0_8_8", \
           "0_65_65", "0_73_73", "0_81_81", "0_87_87", "0_95_95", "0_103_103", "0_111_111", \
           "1_1_1",
        ],
    },
    
    # 1h speakers x 20
    "5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30":{
        # 8+7+20=35mins each
        "train": [
            "0_9_9", "0_10_10", "0_11_11", "0_12_12", "0_13_13", "0_14_14", "0_15_15", "0_16_16", \
            "0_66_66", "0_74_74", "0_82_82", "0_88_88", "0_96_96", "0_104_104", "0_112_112", "0_118_118", \
            "1_2_2", "1_3_3", 
            "1_0_0", "1_4_4", # for speaker 29 only
        ],
        # 4+3.5+5 = 12.5mins each
        # 0_65_a and 0_65_b denote the frist and second half of sequence 0_65_65
        "val": [
            "0_5_5", "0_6_6", "0_7_7", "0_8_8",  \
            "0_65_b", "0_73_b", "0_81_b", "0_87_b", "0_95_b", "0_103_b", "0_111_b", \
            "1_1_b",
        ],
        # 4+3.5+5 = 12.5mins each
        "test": [
           "0_1_1", "0_2_2", "0_3_3", "0_4_4", \
           "0_65_a", "0_73_a", "0_81_a", "0_87_a", "0_95_a", "0_103_a", "0_111_a", \
           "1_1_a",
        ],
    },
}
```
### Other scripts and avatars 
- [Scripts](https://drive.google.com/file/d/1LWzwX2PTqXvL75_RJFbss5nMb0epQug0/view?usp=share_link) for videos in [rendered videos](https://drive.google.com/drive/folders/1ghZ7_4LkCyM_IZxTElzAwPzGheLrBGBu)
- [Annotation tools](https://drive.google.com/file/d/1HuVoTWgDXKX0M3yfei1jY3Pr7ZMF5k3G/view?usp=share_link) for semantic relevancy.
- Avatars in our paper is from HumanGenerator V3, we could share the avatars after confirming your liscense of HGv3 by email. 



# Reproduction
### Train and test CaMN
0. `python == 3.7`
1. build folders like:
    ```
    audio2pose
    ├── codes
    │   └── audio2pose
    ├── datasets
    │   ├── beat_raw_data
    │   ├── beat_annotations
    │   └── beat_cache
    └── outputs
        └── audio2pose
            ├── custom
            └── wandb   
    ```
2. download the scripts to `codes/audio2pose/`
3. run ```pip install -r requirements.txt``` in the path `./codes/audio2pose/`
4. download full dataset to `datasets/beat`
5. bulid data cache and calculate mean and std by given `number of joints`, `FPS`, `speakers` using `/dataloader/preprocessing.ipynb`   
6. `cd ./dataloaders && python build_vocab.py` for language model
7. run ```python train.py -c ./configs/ae_4english_15_141.yaml``` for pretrained_ae for FID calculation, or download [pretrained ckpt](https://drive.google.com/file/d/1kgUramYf8US2pg1lssJBflDqV3_CJpqH/view?usp=share_link) to `/datasets/beat_cache/cache_name/weights/`
8. run ```python train.py -c ./configs/camn_4english_15_141.yaml``` for training or or download [pretrained ckpt](https://drive.google.com/file/d/1Q7v_e3K_cgR93a70hlen7KAf2K94NU7j/view?usp=share_link) to `/datasets/beat_cache/cache_name/weights/`.
9. run ```python test.py -c ./configs/camn_4english_15_141.yaml``` for inference.
10. load ```./outputs/audio2pose/custom/exp_name/epoch_number/xxx.bvh``` into blender to visualize the test results.

### From data in other dataset (e.g. Trinity) 
- refer `train and test CaMN` for bvh cache
- remove modalities, e.g., remove facial expressions.
    - set `facial_rep: None` and `facial_f: 0` in `.yaml`
    - set `dataset: trinity` in `.yaml`

## Citation
BEAT is established for the following research project. Please consider cite our work if you use BEAT dataset.
```bib
@article{liu2022beat,
  title   = {BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis},
  author  = {Haiyang Liu, Zihao Zhu, Naoya Iwamoto, Yichen Peng, Zhengqing Li, You Zhou, Elif Bozkurt, Bo Zheng},
  journal = {European Conference on Computer Vision},
  year    = {2022}
}
```
