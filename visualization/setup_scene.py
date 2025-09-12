"""
Blender script to set up and generate a minimal simple_scene.blend.

Design: The scene only contains a Sun light, a Camera, and a Plane named 'Ground'.
All motion (BVH import) and textures are handled in visualization/visualize_with_texture.py.
"""

import bpy
import os


def setup_and_save_scene(output_path):
    print("--- Setting up minimal base scene (light, camera, plane) ---")

    # 1) Fresh scene
    bpy.ops.wm.read_homefile(use_empty=True)

    # 2) Lighting (Sun)
    print("Creating sun light...")
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, -6, 6))
    light_obj = bpy.context.object
    light_obj.data.energy = 2.0
    light_obj.rotation_euler = (0.9, 0.0, 0.0)

    # 3) Camera
    print("Creating camera...")
    bpy.ops.object.camera_add(align='VIEW', location=(0, -10, 4), rotation=(1.2, 0, 0))
    camera_obj = bpy.context.object
    camera_obj.name = 'Camera'
    bpy.context.scene.camera = camera_obj

    # 4) Ground plane
    print("Creating ground plane...")
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    plane_obj = bpy.context.object
    plane_obj.name = 'Ground'

    # Give the plane a simple material
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    plane_obj.data.materials.append(mat)

    print(f"\nSaving scene to: {output_path}")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
        print("--- Scene setup finished successfully! ---")
    except Exception as e:
        print(f"FATAL: Could not save .blend file. Error: {e}")


if __name__ == "__main__":
    output_blend_file = os.path.join("visualization", "simple_scene.blend")
    setup_and_save_scene(output_blend_file)
