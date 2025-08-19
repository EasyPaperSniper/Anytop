# Project Component: Chain-Motion-Based Embedding for Anytopology Character Motion Generation


## Scope
This sub-project focuses on developing a neural network that predicts a skeletal structure and skinning weights for a given 3D character mesh.

## Key Files
* `mesh_processor.py`: Handles loading and normalizing `OBJ` or `FBX` meshes.
* `rigging_model.py`: The core PyTorch model for rigging.
* `dataset_creator.py`: Script to generate a dataset of rigged characters for training.

## Component-Specific Instructions
* **Task:** The primary task is to write code that prepares mesh data (e.g., voxelization, point clouds) for input to the rigging model.
* **Model Architecture:** The current model is a Graph Neural Network (GNN). All code should assume this architecture. When suggesting improvements, propose GNN-related techniques (e.g., different aggregation methods, attention mechanisms).
* **Data Format:** The output should be a `.skel` file and a `.skin` file. The format for these files is defined in `../docs/file_formats.md`. Adhere to this format strictly.
* **Collaboration:** When reviewing code or suggesting changes, ensure the new code integrates seamlessly with `mesh_processor.py` and `rigging_model.py`.