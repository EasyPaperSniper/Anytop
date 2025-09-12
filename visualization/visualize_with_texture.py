# Blender script for visualizing BVH motion with a custom texture.

import bpy
import sys
import os
import argparse

def clear_animation(armature_name='Armature'):
    """Finds the armature and clears any existing animation data."""
    print(f"Attempting to clear animation from '{armature_name}'...")
    armature = bpy.data.objects.get(armature_name)
    
    if not armature:
        print(f"   WARNING: Armature '{armature_name}' not found.")
        return

    if armature.animation_data and armature.animation_data.action:
        armature.animation_data.action = None
        print("   - Cleared existing action from armature.")

    # Also clear animation from any child mesh objects
    for child in armature.children:
        if child.type == 'MESH' and child.animation_data and child.animation_data.action:
            child.animation_data.action = None
            print(f"   - Cleared existing action from child mesh '{child.name}'.")

def apply_texture(texture_path, armature_name='Armature'):
    """Finds the character mesh and applies the given texture."""
    print(f"Attempting to apply texture '{texture_path}'...")
    armature = bpy.data.objects.get(armature_name)
    if not armature:
        print(f"   WARNING: Armature '{armature_name}' not found. Cannot find child mesh.")
        return

    # Find the first mesh object parented to the armature
    mesh_obj = next((child for child in armature.children if child.type == 'MESH'), None)
    
    if not mesh_obj:
        print("   WARNING: No child mesh found for the armature.")
        return

    if not mesh_obj.active_material:
        print(f"   WARNING: Mesh '{mesh_obj.name}' has no active material.")
        return

    # Find the image texture node in the material
    mat = mesh_obj.active_material
    tex_node = next((node for node in mat.node_tree.nodes if node.type == 'TEX_IMAGE'), None)

    if not tex_node:
        print("   WARNING: No Image Texture node found in the material.")
        return

    # Load the new image and assign it
    try:
        new_image = bpy.data.images.load(texture_path)
        tex_node.image = new_image
        print(f"   - Successfully applied texture to material '{mat.name}'.")
    except Exception as e:
        print(f"   FAIL: Could not load texture file. Make sure path is correct. Error: {e}")

def apply_bvh(bvh_path, armature_name='Armature'):
    """Imports the BVH file and applies it to the armature."""
    print(f"Attempting to apply BVH motion from '{bvh_path}'...")
    armature = bpy.data.objects.get(armature_name)
    if not armature:
        print(f"   WARNING: Armature '{armature_name}' not found. Cannot apply BVH.")
        return

    # Select the armature to make it the target for import
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)

    try:
        # Import the BVH motion. Scale is often needed to match Blender's units.
        bpy.ops.import_anim.bvh(filepath=bvh_path, 
                                axis_forward='-Z', 
                                axis_up='Y', 
                                target='ARMATURE', 
                                global_scale=0.01, # This may need adjustment
                                use_fps_scale=True)
        print("   - Successfully imported and applied BVH motion.")
    except Exception as e:
        print(f"   FAIL: Could not import BVH file. Error: {e}")

def main():
    """Main function to orchestrate the visualization process."""
    # --- Argument Parsing ---
    # Blender scripts require a special way to parse arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Visualize BVH motion with a texture in Blender.")
    parser.add_argument("--bvh_path", required=True, help="Path to the input .bvh file.")
    parser.add_argument("--texture_path", required=True, help="Path to the input .jpg texture file.")
    parser.add_argument("--output_blend", default=None, help="Optional: Path to save the final .blend file.")
    args = parser.parse_args(argv)

    # --- Main Logic ---
    # 1. Load the base scene
    # Note: The script assumes it is run from the project root directory.
    base_scene_path = "visualization/simple_scene.blend"
    if not os.path.exists(base_scene_path):
        print(f"FATAL: Base scene not found at '{base_scene_path}'. Exiting.")
        return
    bpy.ops.wm.open_mainfile(filepath=base_scene_path)
    print(f"Successfully loaded base scene: {base_scene_path}")

    # 2. Clear any previous animation
    clear_animation()

    # 3. Apply the new texture
    apply_texture(args.texture_path)

    # 4. Apply the new BVH motion
    apply_bvh(args.bvh_path)

    # 5. Save the output file if requested
    if args.output_blend:
        try:
            bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)
            print(f"\nSuccessfully saved final scene to: {args.output_blend}")
        except Exception as e:
            print(f"   FAIL: Could not save .blend file. Error: {e}")

    print("\n--- Visualization script finished. ---")

if __name__ == "__main__":
    main()
