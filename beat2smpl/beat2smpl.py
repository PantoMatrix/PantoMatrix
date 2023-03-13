import bpy
import os
MAIN_DIR = "D:\\BEAT_Dataset\\female\\"#!!!!!!!!CHANGE GENDER HERE!!!!!!!!

def purge_orphans():
    if bpy.app.version >= (3, 0, 0):
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    else:
        result = bpy.ops.outliner.orphans_purge()
        if result.pop() != "CANCELLED":
            purge_orphans()

for speaker_id in os.listdir(MAIN_DIR):
    for file in os.listdir(MAIN_DIR+speaker_id):
        if file.endswith(".bvh") and not file.endswith("_genea.bvh"):
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            gender = "female"#!!!!!!!!CHANGE GENDER HERE!!!!!!!!
            fp = MAIN_DIR+speaker_id+"\\"+file
            bpy.ops.import_anim.bvh(filepath=fp)
            bpy.ops.scene.smplx_add_gender()
            scene = bpy.context.scene
            gender_obj = "SMPLX-" + gender
            gender_mesh = "SMPLX-mesh-" + gender
            for obj in scene.objects:
                if obj.name != gender_obj and obj.name != gender_mesh:
                    scene.source_rig = obj.name
                if obj.name == gender_obj:
                    scene.target_rig = obj.name
            bpy.ops.arp.auto_scale()
            bpy.ops.arp.build_bones_list()
            bpy.ops.arp.import_config_preset(filepath="beat2smpl.bmap")
            with open(fp, "r") as pose_data:
                frame_len = 0
                for line in pose_data.readlines():
                    frame_len += 1
                frame_len -= 433
            print(frame_len)
            bpy.ops.arp.retarget('EXEC_DEFAULT', frame_start=0, frame_end=frame_len)
            
            scene.frame_start = 0
            scene.frame_end = frame_len
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[gender_mesh].select_set(True)
            bpy.context.view_layer.objects.active = bpy.data.objects[gender_mesh]
            new_name = MAIN_DIR + speaker_id + "\\" + file[:-4] + ".fbx"
            bpy.ops.object.smplx_export_fbx(filepath=new_name)
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            purge_orphans()