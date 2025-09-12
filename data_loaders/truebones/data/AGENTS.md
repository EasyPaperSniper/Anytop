# Data Loading Scripts

This directory contains the PyTorch Dataset and Sampler classes for loading motion data.

---

### `dataset.py`

This script is the primary data loader for training the main **full-body** motion generation model.

- **`TruebonesDataset`**: The main entry-point class. It wraps the `MotionDataset` and handles configuration, such as selecting a subset of characters to train on (e.g., only bipeds).
- **`MotionDataset`**: The core class that loads, processes, and augments full-body motion data. It handles Z-normalization, padding/trimming sequences, and data augmentation (adding/removing joints).
- **`TruebonesSampler`**: A `WeightedRandomSampler` that ensures the model trains on a balanced distribution of different character types.

---

### `limb_emb_dataset.py`

This script is a specialized data loader for training the **limb-level VQ-VAE** (`LimbVQVAE`). Its purpose is to provide the model with isolated motion windows from individual limbs.

- **`LimbDataset`**: The main entry-point class that wraps `LimbMotionDataset`.
- **`LimbMotionDataset`**: The core class that:
    1. Loads full-body motion data.
    2. Identifies the kinematic chains (limbs) for each character.
    3. Randomly samples a single limb from a motion clip.
    4. Extracts only the 6D rotation data for that limb.
    5. Slices a fixed-length time window from the limb's motion.
    6. Pads the data along the joint dimension to handle limbs with varying numbers of joints and creates a `joint_mask`.
- **`LimbSampler`**: A `WeightedRandomSampler` that ensures balanced training across all character and limb types.

---

### `test_limb_emb_dataset.py`

This is a test script to verify the correctness of `limb_emb_dataset.py`.

**Purpose:** To ensure that the complex data processing (sampling, slicing, padding, re-indexing) is working as expected.

**Checks Performed:**
- **Shape Verification:** Confirms that all output tensors have the correct dimensions.
- **Mask and Padding Correctness:** Ensures that the `joint_mask` is accurate and that the padded areas of the motion tensor are zero.
- **Parent Re-indexing:** Verifies that the parent array for the sampled limb is structurally correct (i.e., has exactly one root joint).
- **Data Integrity:** Performs a sanity check by comparing the output data with the original data from the `.npy` file to make sure no corruption has occurred.

**To Run the Test:**

Execute the script as a module from the root directory of the project:
```bash
python -m data_loaders.truebones.data.test_limb_emb_dataset
```
