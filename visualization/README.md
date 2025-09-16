# Guide: BVH + Textured Plane Visualization in Blender

This guide documents the minimal, robust pipeline for visualizing a BVH animation in Blender while displaying an optional image sequence as a texture on a ground plane. The scene template is intentionally simple and motion‑agnostic; all motion logic happens in the visualization script.

---

## 1. Design Principles

1) Minimal base scene: The template scene contains only a light, a camera, and a plane named `Ground`. No armature or character lives in the template.
2) Clean state per run: The visualization script removes everything except the light(s), camera(s), and the `Ground` plane; then it imports the BVH to create a fresh armature and animation.
3) Optional texture as image sequence: A directory of numbered images (jpg/png) can be applied to the `Ground` plane as an animated texture. If no directory is given, the plane keeps its default material.
4) Correct timing: The scene FPS can be set (default 20), and the frame range is aligned to the BVH action range.

## 2. Workflow Overview

1) Load the base scene from `visualization/simple_scene.blend` (contains light, camera, plane).
2) Purge everything not in the base set (lights, cameras, and the `Ground` plane).
3) Import the BVH using Blender UI defaults (orientation preserved): `axis_forward='-Z'`, `axis_up='Y'`, `target='ARMATURE'`, `global_scale=1.0`. This mirrors File → Import → BVH behavior so the character points forward instead of upward.
4) Set scene FPS (default 20) and align frame range to the imported action.
5) If `--texture_dir` is provided, create a plane and parent it to the root bone of the imported armature. Apply the folder of images as an image‑sequence material to that plane so it moves with the skeleton.

## 3. Implementation Details & Assumptions

The scripts rely on Blender’s Python API (`bpy`).

Base scene (`setup_scene.py`) guarantees objects named:
- Plane mesh named `Ground`.
- At least one Sun light.
- A Camera.

Visualization (`visualize_with_texture.py`) assumptions:
- Keeps only light(s), camera(s), and the mesh named `Ground`; deletes others before import.
- BVH import creates a new armature and links an Action automatically.
- Texture directory, if provided, contains numbered images for an image‑sequence texture; if not numbered, only the first image may display.

## 4. Generating the Template Scene (`setup_scene.py`)

### Purpose

Create and save a minimal `.blend` scene containing:
- A Sun light.
- A Camera.
- A Plane mesh named `Ground` centered at the origin.

### Command
```bash
blender -b -P visualization/setup_scene.py
```
This creates/overwrites `visualization/simple_scene.blend`.

---

## 5. Using the Visualization Script

Minimal test (no texture), matching Blender UI defaults:
```bash
blender -b -P visualization/visualize_with_texture.py -- \
  --bvh_path dataset/Truebones_processed/bvhs/Raptor2___Run_693.bvh \
  --output_blend visualization/simple_scene.blend \
  --output_mp4 visualization/Raptor2___Run_693.mp4 \
  --fps 20
```



Attach FBX meshes (scaled) to the imported BVH armature (retargeted, keep FBX skinning):
```bash
blender -b -P visualization/visualize_with_texture.py -- \
  --bvh_path dataset/Truebones_processed/bvhs/Raptor2___Run_693.bvh \
  --fbx_path dataset/Truebones_raw/Raptor2/Raptor-TPOSE.fbx \
  --fbx_scale 1.0 \
  --fbx_use_manual_orientation --fbx_axis_forward Y --fbx_axis_up Z \
  --source_arm_cleanup hide \
  --output_blend visualization/simple_scene.blend \
  --output_mp4 visualization/Raptor2___Run_693_fbx.mp4 \
  --fps 20
```


Notes on FBX meshes and retargeting
- The script imports the FBX and keeps the first imported FBX armature plus its meshes; other imported object types are removed. It uniformly scales the FBX armature and meshes (without applying transforms).
- If both armatures exist (BVH and FBX) and meshes were imported, the script retargets: it adds Copy Rotation constraints per bone (and Copy Location for the root) from the BVH armature to the FBX armature, then bakes the result to an action on the FBX armature and clears constraints. The meshes keep their original skinning and follow the baked motion.
- If no FBX armature is present, the script falls back to binding the meshes to the BVH armature with automatic weights.

FBX orientation control
- The script does not apply any initial rotation to the imported FBX. Use FBX manual orientation flags at import time if needed: `--fbx_use_manual_orientation --fbx_axis_forward mZ --fbx_axis_up Y` (mX/mY/mZ represent negative axes).

Z motion handling
- Root translation X, Y, and Z are copied directly from the BVH root to the FBX root (world space). No additional Z scaling is applied.

Source armature cleanup
- Control the BVH source armature after bake with `--source_arm_cleanup {keep,hide,delete}`. Example uses `hide` to keep the scene clean while preserving the source for inspection.

Notes
- The script overwrites the `.blend` given by `--output_blend`.
- The MP4 is saved to the given path; parent directories are created if missing.
- Image sequence works best when files are numerically indexed (e.g., tex_0001.jpg, tex_0002.jpg, …). The plane is parented to the root bone and follows the motion.
If your manual UI import used non-default axes, we can expose axis overrides, but by default the script now mirrors Blender’s standard BVH import (Forward -Z, Up Y).
