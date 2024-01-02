# Retarget BEAT bvh to SMPL fbx

### Prerequisite
0. `Blender Version == 2.93`
1. `Blender Add-on : Auto-Rig Pro` if you do not have ARP, please purchase [here](https://blendermarket.com/products/auto-rig-pro).
2. `Blender Add-on : SMPL-X Blender Add-on (300 shape components, 20220623, 310MB)` if you do not have SMPL-X, please download [here（Registration is needed.）](https://smpl-x.is.tue.mpg.de/index.html).

### Data Description
0. Data should be put in this hierachy:
    ```
    BEAT_Dataset
    ├── female
    │   └── FEMALE_ID
    │       └── MOTION_FILES.bvh
    └── male
        └── MALE_ID
            └── MOTION_FILES.bvh  
    ```
1. FEMALE_IDs are `6-10, 21-30`. MALE_IDs are `1-5, 11-20`. SMPL-X Blender Add-on is defaultly set to `Female`, please manually set to `Male` on `SMPL-X Model` panel in the sidebar of Blender viewer, if you want to process `Male` data. 
2. Please delete `.0` in end of line `430` in `30_katya_1_1_1.bvh`, `30_katya_1_2_2.bvh`, `30_katya_1_3_3.bvh`, `14_zhang_1_2_2.bvh`, `14_zhang_1_3_3.bvh` before run the processing code.

### Processing
0. Install `Auto-Rig Pro` and `SMPL-X for Blender` in `Blender 2.93`.
1. Put remapping index `./beat2smpl/beat2smpl.bmap` in the scripts folder of Blender. (ex. In Windows, this folder are in `C:\Users\YOUR_NAME\AppData\Roaming\Blender Foundation\Blender\2.93\scripts\addons\auto_rig_pro-master\remap_presets\`.)
2. Add `fp: StringProperty(subtype="FILE_PATH", default='./smpl.fbx')` in line `1125` of `__init__.py` in SMPL-X Blender Add-on installed path. (ex. In Windows, this folder are in `C:\Users\YOUR_NAME\AppData\Roaming\Blender Foundation\Blender\2.93\scripts\addons\smplx_blender_addon\__init__.py`.)
3. Read `./beat2smpl/beat2smpl.py` into Blender script, and turn on `Window/Toggle System Console` in Blender for checking the progress of retargeting.
4. Change the gender in both line `3` and `20` in `./beat2smpl/beat2smpl.py`,  and run the script in Blender.
