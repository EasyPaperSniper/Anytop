import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from os.path import join as pjoin
from data_loaders.truebones.truebones_utils.get_opt import get_opt


def list_characters_kinematic_chains(objects_subset: Optional[str] = None,
                                     verbose: bool = True,
                                     save_json: bool = False,
                                     json_path: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    List all characters in the Truebones condition file and their kinematic chains,
    including both joint indices and joint names for each chain.

    Parameters
    - objects_subset: Optional name of a predefined subset as defined by
      `opt.subsets_dict` (e.g., "all", "quadropeds", "bipeds"). If None, uses all
      available characters found in the condition dictionary.
    - verbose: If True, prints a readable summary to stdout.

    Returns
    - A dictionary mapping `character_name` -> list of chain descriptors. Each chain
      descriptor is a dict with keys:
        - 'chain_index': int
        - 'joint_indices': List[int]
        - 'joint_names': List[Optional[str]]
        - 'pairs': List[Tuple[int, Optional[str]]]  # (index, name)

    Example
    >>> out = list_characters_kinematic_chains(objects_subset="bipeds", verbose=False,
    ...                                        save_json=True)
    >>> list(out.keys())[:2]
    ['Ostrich', 'Flamingo']
    >>> first = out['Ostrich'][0]
    >>> isinstance(first['joint_indices'], list) and isinstance(first['pairs'][0], tuple)
    True
    """
    # Resolve dataset paths using the same logic as dataset loaders
    opt = get_opt(None)
    opt.data_root = pjoin('.', opt.data_root)
    opt.motion_dir = pjoin('.', opt.motion_dir)
    cond_dict: Dict[str, Dict[str, Any]] = np.load(opt.cond_file, allow_pickle=True).item()

    # Optional subsetting of characters
    if objects_subset is not None:
        subset = opt.subsets_dict.get(objects_subset)
        if subset is None:
            raise ValueError(f"Unknown objects_subset '{objects_subset}'. "
                             f"Available: {list(opt.subsets_dict.keys())}")
        cond_dict = {k: cond_dict[k] for k in subset if k in cond_dict}

    result: Dict[str, List[Dict[str, Any]]] = {}

    pretty_lines: List[str] = []

    for object_type, conditions in cond_dict.items():
        # Extract and normalize chains to plain Python lists
        raw_chains = conditions.get('kinematic_chains', [])
        if isinstance(raw_chains, np.ndarray):
            raw_chains = raw_chains.tolist()
        chains: List[List[int]] = []
        for c in (raw_chains or []):
            if isinstance(c, np.ndarray):
                chains.append(c.astype(int).tolist())
            else:
                chains.append([int(x) for x in c])

        # Joint names as a list (may be shorter than max index; fill with None)
        joints_names = conditions.get('joints_names')
        if isinstance(joints_names, np.ndarray):
            joints_names = joints_names.tolist()
        if joints_names is None:
            joints_names = []

        # Parent count (may be ndarray or list)
        parents = conditions.get('parents')
        if isinstance(parents, np.ndarray):
            num_parents = int(parents.shape[0])
        elif parents is None:
            num_parents = 0
        else:
            num_parents = len(parents)

        character_out: List[Dict[str, Any]] = []
        for ci, chain in enumerate(chains):
            names_for_chain: List[Optional[str]] = [
                (joints_names[j] if j < len(joints_names) else None) for j in chain
            ]
            pairs: List[Tuple[int, Optional[str]]] = list(zip(chain, names_for_chain))
            character_out.append({
                'chain_index': ci,
                'joint_indices': list(chain),
                'joint_names': names_for_chain,
                'pairs': pairs,
            })

        result[object_type] = character_out

        header = f"Character: {object_type} | J={num_parents} | Chains={len(chains)}"
        if verbose:
            print(header)
        pretty_lines.append(header)
        if len(chains) == 0:
            if verbose:
                print("  (no kinematic_chains defined)\n")
            pretty_lines.append("  (no kinematic_chains defined)")
            pretty_lines.append("")
            continue
        for entry in character_out:
            ci = entry['chain_index']
            pairs_str = ', '.join([
                f"{idx}:{name if name is not None else 'None'}" for idx, name in entry['pairs']
            ])
            line = f"  - Chain {ci:02d} (len={len(entry['joint_indices'])}): [ {pairs_str} ]"
            if verbose:
                print(line)
            pretty_lines.append(line)
        if verbose:
            print()
        pretty_lines.append("")

    # Optionally save to JSON
    if save_json:
        out_path = json_path or 'kinematic_chain.json'
        # Ensure directory exists if a path with dirs is provided
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # Programmatic JSON output (unchanged)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        # Human-readable companion file (pretty text)
        pretty_path = os.path.splitext(out_path)[0] + '.txt'
        with open(pretty_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(pretty_lines).rstrip() + '\n')
        if verbose:
            print(f"Saved kinematic chains JSON to: {out_path}")
            print(f"Saved human-readable chains to: {pretty_path}")

    return result


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
    # Local imports to avoid hard dependency when only listing chains
    import torch
    from data_loaders.truebones.data.limb_emb_dataset import LimbDataset

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
    # test_limb_dataset()
    list_characters_kinematic_chains()
