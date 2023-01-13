# Data Verson

### [v0.2.1](https://drive.google.com/file/d/1Akf0WgAwuH2fvlWbvNpif4XRqXlpznh9/view?usp=share_link) (English)
- fix some incorrect audio-motion sync. Now all data are with correct sync.
    - all sequences for speaker 1 and 5
    - `12_zhao_0_9_16`
- fix the orientation issue for default T-Pose. Now all speakers' T-Pose are in the same orientation.
- set the root translation in the first frame as `0, 0, 0` for all speakers.
- updated correct data for `4_lawrence_0_75_75` and  `4_lawrence_0_76_76`. 
- updated the missing textgrid for `7_sophie_1_3_3`, `7_sophie_1_4_4`, `7_sophie_1_9_9`, `7_sophie_1_10_10`.

### v0.2.0 (English: beat_rawdata_english and beat_annotation_english)
- updated emotional and semantic annotation for English data
- cut sequences by 1 min for speech sections, 10 mins for conversation sections. 

### [v0.1.0](https://drive.google.com/drive/folders/1CVyJOp3G_A9l1N_CsKdHgXQfB4pXhG8c?usp=share_link) (All languages)
- audio, facial, motion data for all languages 
- text for English 
- rendered videos for all languages 


# Full News List 
- **[Ongoing]** SMPL-X version data.
- **[2023/01]** [English data v0.2.1](https://drive.google.com/file/d/1Akf0WgAwuH2fvlWbvNpif4XRqXlpznh9/view?usp=share_link) are available. Fix the orientation issue. See [updation](./docs/updation.md) for details.
- **[2023/01]** Provide checkpoints (#14, #16), scripts for rendering (#17), preprocessing (#18).  
- **[2022/12]** Provide English data v0.2.0 in Zip files (#10).
- **[2022/10]** [Project page](https://pantomatrix.github.io/BEAT/) and [rendered videos](https://drive.google.com/drive/folders/1ghZ7_4LkCyM_IZxTElzAwPzGheLrBGBu) are available.
- **[2022/08]** [All languages data v0.1.0](https://drive.google.com/drive/folders/1CVyJOp3G_A9l1N_CsKdHgXQfB4pXhG8c?usp=share_link)  (in separated files) are available.
- **[2022/03]** CaMN training scripts from [anonymous submission](https://github.com/beat2022dataset/beat).
