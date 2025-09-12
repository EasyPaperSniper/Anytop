"""Blender script for visualizing a BVH with optional texture sequence on a ground plane.

Design:
- Base scene contains only light(s), camera(s), and a plane named 'Ground'.
- This script purges any other objects, then imports BVH exactly like Blender
  File → Import → BVH defaults (orientation), creating a new armature.
- It aligns the frame range to the BVH action, and optionally applies a
  directory of images as an animated texture on the Ground plane.
"""

import bpy
import sys
import os
import argparse


ALLOWED_KEEP = {"LIGHT", "CAMERA", "MESH"}
GROUND_NAME = "Ground"


def ensure_dir(path: str):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def purge_scene_except_basics():
    """Delete all objects except lights, cameras, and the plane named 'Ground'."""
    to_delete = []
    for obj in list(bpy.data.objects):
        # Keep lights, cameras, and the Ground plane
        if obj.type == 'MESH' and obj.name == GROUND_NAME:
            continue
        if obj.type in {"LIGHT", "CAMERA"}:
            continue
        to_delete.append(obj)
    if not to_delete:
        return
    bpy.ops.object.select_all(action='DESELECT')
    for obj in to_delete:
        try:
            obj.select_set(True)
        except Exception:
            pass
    try:
        bpy.ops.object.delete()
    except Exception as e:
        print(f"   WARNING: Failed to delete some objects: {e}")


def set_scene_fps(fps: int = 20):
    bpy.context.scene.render.fps = int(fps)
    print(f"   - Scene FPS set to {fps}")


def import_bvh_blender_default(bvh_path: str):
    """Import BVH using Blender UI default axis mapping to preserve orientation.

    Mirrors File → Import → BVH defaults:
      - axis_forward='-Z'
      - axis_up='Y'
      - target='ARMATURE' (create a new armature with animation)
      - global_scale=1.0
      - use_fps_scale=True (scale animation timing to scene FPS)
    """
    print(f"Importing BVH (Blender defaults): {bvh_path}")
    before = set(obj.name for obj in bpy.data.objects if obj.type == 'ARMATURE')
    try:
        bpy.ops.import_anim.bvh(
            filepath=bvh_path,
            axis_forward='-Z',
            axis_up='Y',
            target='ARMATURE',
            global_scale=1.0,
            use_fps_scale=True,
        )
    except Exception as e:
        print(f"   FAIL: Could not import BVH: {e}")
        return None
    after = {obj.name for obj in bpy.data.objects if obj.type == 'ARMATURE'}
    new_names = list(after - before)
    if not new_names:
        print("   WARNING: No new armature found after BVH import.")
        return None
    armature = bpy.data.objects[new_names[-1]]
    print(f"   - Imported armature: {armature.name}")
    return armature


def set_frame_range_from_action(armature_obj):
    scene = bpy.context.scene
    action = None
    if armature_obj and armature_obj.animation_data and armature_obj.animation_data.action:
        action = armature_obj.animation_data.action
    if not action and bpy.data.actions:
        action = bpy.data.actions[-1]
    if action:
        scene.frame_start = int(action.frame_range[0])
        scene.frame_end = int(action.frame_range[1])
        print(f"   - Frame range set: {scene.frame_start}..{scene.frame_end}")
    else:
        print("   WARNING: No action found to derive frame range.")


def create_sequence_material(texture_dir: str, frame_start: int, frame_end: int, name: str = "SequenceMaterial"):
    print(f"Creating image sequence material from: {texture_dir}")
    files = [f for f in os.listdir(texture_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print("   WARNING: No images found in texture_dir.")
        return None
    files.sort()
    first_path = os.path.join(texture_dir, files[0])

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    tex = nodes.new(type='ShaderNodeTexImage')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    out = nodes.new(type='ShaderNodeOutputMaterial')
    tex.location = (-400, 300)
    bsdf.location = (-100, 300)
    out.location = (150, 300)
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    try:
        img = bpy.data.images.load(first_path, check_existing=True)
    except Exception as e:
        print(f"   FAIL: Could not load first image: {e}")
        return None
    img.source = 'SEQUENCE'
    tex.image = img
    user = tex.image_user
    user.frame_start = int(frame_start)
    user.frame_duration = max(1, int(frame_end - frame_start + 1))
    user.use_auto_refresh = True
    return mat


def attach_textured_plane_to_skeleton(armature_obj, mat, size: float = 2.0):
    """Create a vertical plane, apply material, and parent it to the armature root bone."""
    if armature_obj is None:
        print("   WARNING: No armature to attach textured plane to.")
        return None
    # Find root bone (bone without parent)
    root_bones = [b for b in armature_obj.data.bones if b.parent is None]
    root_name = root_bones[0].name if root_bones else armature_obj.data.bones[0].name

    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 1.0))
    plane = bpy.context.object
    plane.name = "SkeletonTexturePlane"
    # Rotate plane to face camera (X up, Z forward assumption after import)
    plane.rotation_euler = (0.0, 0.0, 0.0)

    # Apply material
    if mat:
        if not plane.data.materials:
            plane.data.materials.append(mat)
        else:
            plane.data.materials[0] = mat

    # Parent plane to root bone so it follows motion
    bpy.ops.object.select_all(action='DESELECT')
    plane.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    try:
        plane.parent = armature_obj
        plane.parent_type = 'BONE'
        plane.parent_bone = root_name
        print(f"   - Parent plane to armature bone: {root_name}")
    except Exception as e:
        print(f"   WARNING: Bone parenting failed, falling back to object parent: {e}")
        plane.parent = armature_obj
        plane.parent_type = 'OBJECT'
    return plane


def render_animation(output_path: str):
    print(f"Rendering to: {output_path}")
    ensure_dir(output_path)
    scene = bpy.context.scene
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    bpy.ops.render.render(animation=True)
    print("--- Render Finished ---")


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Visualize BVH with optional plane texture sequence.")
    parser.add_argument("--bvh_path", required=True, help="Path to the input .bvh file.")
    parser.add_argument("--texture_dir", default=None, help="Optional: path to a directory of numbered images.")
    parser.add_argument("--output_blend", default=None, help="Optional: save final .blend to this path.")
    parser.add_argument("--output_mp4", default=None, help="Optional: render animation to this MP4 path.")
    parser.add_argument("--fps", type=int, default=20, help="Scene FPS; BVH timing scales to this (default 20).")
    parser.add_argument("--fbx_path", default=None, help="Optional: FBX file that contains target meshes + armature to retarget onto.")
    parser.add_argument("--fbx_scale", type=float, default=1.0, help="Uniform scale to apply to imported FBX meshes before binding.")
    parser.add_argument("--fbx_use_manual_orientation", action='store_true', help="Use manual orientation options when importing FBX.")
    # Use 'mX/mY/mZ' for negative axes to avoid CLI ambiguity with values like '-Z'
    parser.add_argument("--fbx_axis_forward", default='mZ', choices=['X','Y','Z','mX','mY','mZ'], help="FBX import forward axis (X,Y,Z or mX,mY,mZ). Used if --fbx_use_manual_orientation.")
    parser.add_argument("--fbx_axis_up", default='Y', choices=['X','Y','Z','mX','mY','mZ'], help="FBX import up axis (X,Y,Z or mX,mY,mZ). Used if --fbx_use_manual_orientation.")
    # Z motion scaling removed: Z now copied directly from source
    # Cleanup of the source (BVH) armature after bake
    parser.add_argument("--source_arm_cleanup", default='keep', choices=['keep','hide','delete'], help="What to do with the BVH source armature after bake (default keep).")
    # No initial rotation/alignment preset; FBX is used as-imported
    args = parser.parse_args(argv)

    base_scene_path = "visualization/simple_scene.blend"
    if not os.path.exists(base_scene_path):
        print(f"FATAL: Base scene not found at '{base_scene_path}'.")
        return
    bpy.ops.wm.open_mainfile(filepath=base_scene_path)
    print(f"Loaded base scene: {base_scene_path}")

    set_scene_fps(args.fps)

    # Clean the scene to basics
    purge_scene_except_basics()

    # Import BVH (creates a new armature with action)
    armature = import_bvh_blender_default(args.bvh_path)
    if armature:
        set_frame_range_from_action(armature)

    # Optional: attach texture sequence to the skeleton via a plane parented to root bone
    if args.texture_dir and os.path.isdir(args.texture_dir):
        s = bpy.context.scene
        mat = create_sequence_material(args.texture_dir, s.frame_start, s.frame_end, name="SkeletonSeqMat")
        attach_textured_plane_to_skeleton(armature, mat)

    # Optional: import FBX meshes, scale, and bind to the imported BVH armature via automatic weights
    if args.fbx_path and os.path.exists(args.fbx_path):
        print(f"Importing FBX meshes from: {args.fbx_path}")
        before = set(o.name for o in bpy.data.objects)
        try:
            # Prefer importing meshes only; skip animations when supported by this Blender version
            common_kwargs = dict(filepath=args.fbx_path, automatic_bone_orientation=True)
            def _axis_token(a: str) -> str:
                a = a.strip()
                return ('-' + a[1].upper()) if a.lower().startswith('m') and len(a) == 2 else a.upper()
            if args.fbx_use_manual_orientation:
                common_kwargs.update(dict(use_manual_orientation=True,
                                          axis_forward=_axis_token(args.fbx_axis_forward),
                                          axis_up=_axis_token(args.fbx_axis_up)))
            try:
                bpy.ops.import_scene.fbx(**{**common_kwargs, 'use_anim': False})
            except TypeError:
                bpy.ops.import_scene.fbx(**common_kwargs)
        except Exception as e:
            print(f"   FAIL: Could not import FBX: {e}")
        after = set(o.name for o in bpy.data.objects)
        new_objs = [bpy.data.objects[n] for n in (after - before)]
        # Split imported objects and keep first armature for retargeting
        imported_meshes = [o for o in new_objs if o.type == 'MESH']
        imported_armatures = [o for o in new_objs if o.type == 'ARMATURE']
        fbx_arm = imported_armatures[0] if imported_armatures else None
        # Delete any other imported non-essential objects (lights/cameras/empties, extra armatures)
        to_delete = [o for o in new_objs if (o not in imported_meshes and o is not fbx_arm)]
        if to_delete:
            bpy.ops.object.select_all(action='DESELECT')
            for a in to_delete:
                try:
                    a.select_set(True)
                except Exception:
                    pass
            try:
                bpy.ops.object.delete()
            except Exception as e:
                print(f"   WARNING: Could not delete some non-mesh FBX objects: {e}")

        # Apply uniform scale to FBX armature + meshes without applying transforms
        if args.fbx_scale != 1.0:
            if fbx_arm:
                fbx_arm.scale = tuple(s * args.fbx_scale for s in fbx_arm.scale)
            for m in imported_meshes:
                m.scale = tuple(s * args.fbx_scale for s in m.scale)

        # If both armatures exist, retarget animation from BVH armature to FBX armature
        if armature and fbx_arm and imported_meshes:
            print(f"Retargeting motion from '{armature.name}' to FBX armature '{fbx_arm.name}' ...")

            # Clear any existing animation on FBX armature
            if fbx_arm.animation_data and fbx_arm.animation_data.action:
                fbx_arm.animation_data_clear()

            # Add copy constraints per bone
            src_pose_bones = {b.name: b for b in armature.pose.bones}
            for b in fbx_arm.pose.bones:
                src = src_pose_bones.get(b.name)
                if not src:
                    continue
                c_rot = b.constraints.new(type='COPY_ROTATION')
                c_rot.target = armature
                c_rot.subtarget = b.name
                c_rot.target_space = 'POSE'
                c_rot.owner_space = 'POSE'
                if b.parent is None:
                    # Copy root world-space translation (X, Y, Z) from BVH to FBX
                    c_loc = b.constraints.new(type='COPY_LOCATION')
                    c_loc.target = armature
                    c_loc.subtarget = b.name
                    c_loc.target_space = 'WORLD'
                    c_loc.owner_space = 'WORLD'
                    c_loc.use_x = True
                    c_loc.use_y = True
                    c_loc.use_z = True

            # Bake onto FBX armature and clear constraints
            s = bpy.context.scene
            bpy.ops.object.select_all(action='DESELECT')
            fbx_arm.select_set(True)
            bpy.context.view_layer.objects.active = fbx_arm
            bpy.ops.nla.bake(frame_start=int(s.frame_start), frame_end=int(s.frame_end),
                             only_selected=False, visual_keying=True, clear_constraints=True,
                             use_current_action=True, bake_types={'POSE'})
            print("   - Retarget bake completed.")
            # Cleanup source BVH armature per user preference
            try:
                if args.source_arm_cleanup == 'hide':
                    armature.hide_set(True)
                    if hasattr(armature, 'hide_render'):
                        armature.hide_render = True
                    print("   - Source BVH armature hidden.")
                elif args.source_arm_cleanup == 'delete':
                    bpy.ops.object.select_all(action='DESELECT')
                    armature.select_set(True)
                    bpy.context.view_layer.objects.active = armature
                    bpy.ops.object.delete()
                    print("   - Source BVH armature deleted.")
            except Exception as e:
                print(f"   WARNING: Source armature cleanup failed: {e}")
        elif imported_meshes and armature:
            # Fallback to auto-bind if no FBX armature
            for m in imported_meshes:
                try:
                    bpy.ops.object.select_all(action='DESELECT')
                    m.select_set(True)
                    armature.select_set(True)
                    bpy.context.view_layer.objects.active = armature
                    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
                    print(f"   - Bound mesh '{m.name}' to armature '{armature.name}' with auto weights")
                except Exception as e:
                    print(f"   WARNING: Failed to bind mesh '{m.name}': {e}")

    # Save .blend
    if args.output_blend:
        ensure_dir(args.output_blend)
        try:
            bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)
            print(f"Saved final scene to: {args.output_blend}")
        except Exception as e:
            print(f"   FAIL: Could not save .blend: {e}")

    # Render MP4
    if args.output_mp4:
        render_animation(args.output_mp4)

    print("\n--- Visualization script finished. ---")


if __name__ == "__main__":
    main()
