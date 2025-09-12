import torch
import numpy as np
from data_loaders.truebones.data.limb_emb_dataset import LimbDataset


def test_limb_dataset():
    """
    This test function verifies the integrity of the LimbDataset.
    It checks for:
    1. Correct output shapes.
    2. Correctness of the joint mask.
    3. Correctness of padding.
    4. Validity of the re-indexed parent array for the limb.
    5. Data integrity by comparing a slice of the output to the original data.
    """
    print("--- Running Test for LimbDataset ---")

    # 1. Instantiate the Dataset
    print("1. Instantiating LimbDataset...")
    try:
        dataset = LimbDataset(split="train", num_frames=64)
        assert len(dataset) > 0
        print("   PASS: Dataset instantiated successfully.")
    except Exception as e:
        print(f"   FAIL: Could not instantiate dataset. Error: {e}")
        return

    # 2. Get a single sample
    print("\n2. Fetching a single sample...")
    try:
        # We need to access the inner dataset to get some original data for comparison
        motion_dataset = dataset.motion_dataset
        sample_idx = random.randint(0, len(motion_dataset) - 1)
        
        padded_limb_motion, joint_mask, local_parents, padded_bone_lengths, motion_length = motion_dataset[sample_idx]
        print(f"   PASS: Sample {sample_idx} fetched successfully.")
    except Exception as e:
        print(f"   FAIL: Could not fetch sample. Error: {e}")
        return

    # 3. Perform Verification Checks
    print("\n3. Performing verification checks...")
    
    # Get original data for comparison
    motion_name = motion_dataset.motion_clips[sample_idx]
    limb_key = motion_dataset.limb_keys[sample_idx]
    object_type, limb_idx_str = limb_key.split('_')
    limb_idx = int(limb_idx_str)
    conditions = motion_dataset.cond_dict[object_type]
    limb_chain = conditions['kinematic_chains'][limb_idx]
    num_limb_joints = len(limb_chain)

    # Check Shapes
    try:
        assert padded_limb_motion.shape == (64, motion_dataset.max_limb_joints, 6)
        assert joint_mask.shape == (motion_dataset.max_limb_joints,)
        assert local_parents.shape == (motion_dataset.max_limb_joints,)
        assert padded_bone_lengths.shape == (motion_dataset.max_limb_joints, 1)
        print("   PASS: All output shapes are correct.")
    except AssertionError:
        print("   FAIL: Shape mismatch detected.")
        return

    # Check Mask Correctness
    try:
        assert int(torch.sum(joint_mask).item()) == num_limb_joints
        print("   PASS: Joint mask has the correct number of valid joints.")
    except AssertionError:
        print("   FAIL: Joint mask sum does not match number of limb joints.")
        return

    # Check Padding Correctness
    try:
        padded_area = padded_limb_motion[:, num_limb_joints:, :]
        assert torch.all(padded_area == 0)
        print("   PASS: Padded area of the motion tensor is all zeros.")
    except AssertionError:
        print("   FAIL: Padded area contains non-zero values.")
        return

    # Check Parent Re-indexing
    try:
        valid_parents = local_parents[:num_limb_joints]
        assert torch.sum(valid_parents == -1) == 1
        print("   PASS: Re-indexed parents array has exactly one root (-1).")
    except AssertionError:
        print("   FAIL: Incorrect number of roots in parent array.")
        return

    # Check Data Integrity
    try:
        full_motion = np.load(pjoin(motion_dataset.opt.motion_dir, motion_name))
        original_6d_data = full_motion[:, limb_chain, 3:9]
        
        # Take a slice from the original data that corresponds to the output
        # Note: The test can't know the random start_idx, so we just check the first frame
        # if the motion was shorter than the window.
        if motion_length < 64:
            original_slice = torch.from_numpy(original_6d_data).float()
            output_slice = padded_limb_motion[:motion_length, :num_limb_joints, :]
            assert torch.allclose(original_slice, output_slice)
            print("   PASS: Data integrity check passed (compared with original .npy file).")
        else:
            print("   INFO: Data integrity check skipped for long motion (random slicing).")

    except Exception as e:
        print(f"   FAIL: Data integrity check failed. Error: {e}")
        return
        
    print("\n--- All Tests Passed Successfully! ---")

if __name__ == '__main__':
    import random
    from os.path import join as pjoin
    # To run this test, you must be in the root directory of the project.
    # Example: python -m data_loaders.truebones.data.test_limb_emb_dataset
    test_limb_dataset()