# Blender script to set up and generate the simple_scene.blend file.

import bpy
import os

def setup_and_save_scene(output_path):
    """
    Creates a complete, simple scene with camera, lighting, and a character rig,
    then saves it as a .blend file.
    """
    print("--- Setting up new scene --- ")

    # 1. Start with a fresh, empty scene
    bpy.ops.wm.read_homefile(use_empty=True)

    # 2. Set up lighting
    print("Creating light source...")
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 0))
    light_obj = bpy.context.object
    light_obj.location = (0, -5, 5)
    light_obj.rotation_euler = (0.785398, 0, 0) # 45 degrees

    # 3. Set up camera
    print("Creating camera...")
    bpy.ops.object.camera_add(align='VIEW', location=(0, -8, 2), rotation=(1.3, 0, 0))
    camera_obj = bpy.context.object
    camera_obj.name = 'Camera'

    # 4. Create the Armature
    print("Creating armature...")
    bpy.ops.object.armature_add_one(enter_editmode=False, location=(0, 0, 0))
    armature_obj = bpy.context.object
    armature_obj.name = 'Armature'

    # 5. Create the Character Mesh
    print("Creating character mesh...")
    # Using a UV sphere as a simple placeholder mesh
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 1))
    mesh_obj = bpy.context.object
    mesh_obj.name = 'CharacterMesh'
    
    # Parent the mesh to the armature
    mesh_obj.parent = armature_obj
    
    # Add an armature deform modifier so the mesh will follow the skeleton
    modifier = mesh_obj.modifiers.new(name='ArmatureDeform', type='ARMATURE')
    modifier.object = armature_obj
    print(f"   - Parented '{mesh_obj.name}' to '{armature_obj.name}'.")

    # 6. Create the Material with a placeholder Texture Node
    print("Creating material and texture node...")
    # Create a new material
    mat = bpy.data.materials.new(name="CharacterMaterial")
    mat.use_nodes = True
    mesh_obj.data.materials.append(mat)
    
    # Get the node tree and clear default nodes
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create the necessary nodes
    node_texture = nodes.new(type='ShaderNodeTexImage')
    node_texture.name = 'Image Texture' # Important name for the other script
    node_texture.location = (-300, 300)
    
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (0, 300)

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (300, 300)

    # Link the nodes together: Image Texture -> BSDF -> Output
    links = mat.node_tree.links
    links.new(node_texture.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    print("   - Material setup complete.")

    # 7. Save the final .blend file
    print(f"\nSaving scene to: {output_path}")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
        print("--- Scene setup finished successfully! ---")
    except Exception as e:
        print(f"FATAL: Could not save .blend file. Error: {e}")

if __name__ == "__main__":
    # The output path for the generated scene file
    # Assumes the script is run from the project root directory
    output_blend_file = os.path.join("visualization", "simple_scene.blend")
    
    setup_and_save_scene(output_blend_file)
