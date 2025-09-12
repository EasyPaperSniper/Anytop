blender -b -P visualization/visualize_with_texture.py -- \
  --bvh_path dataset/Truebones_processed/bvhs/Raptor2___Run_693.bvh \
  --fbx_path dataset/Truebones_raw/Raptor2/Raptor-TPOSE.fbx \
  --fbx_scale 1 \
  --fbx_use_manual_orientation --fbx_axis_forward Y --fbx_axis_up Z \
  --z_scale_mode auto \
  --source_arm_cleanup hide \
  --output_blend visualization/simple_scene.blend \
  --fps 20
