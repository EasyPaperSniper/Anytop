import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
from os.path import join as pjoin
import random
from torch.utils.data._utils.collate import default_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt

def collate_fn(batch):
    """This function is not strictly necessary for this dataset but good practice."""
    batch.sort(key=lambda x: x[1], reverse=True) # Sort by motion length
    return default_collate(batch)

class LimbMotionDataset(data.Dataset):
    def __init__(self, opt, cond_dict, max_limb_joints=20):
        self.opt = opt
        self.cond_dict = cond_dict
        self.max_motion_length = opt.max_motion_length
        self.max_limb_joints = max_limb_joints
        
        self.motion_clips = []
        self.limb_keys = []

        for object_type, conditions in self.cond_dict.items():
            kinematic_chains = conditions.get('kinematic_chains', [])
            if not kinematic_chains:
                continue

            object_motions = [f for f in os.listdir(opt.motion_dir) if f.startswith(f'{object_type}_')]
            
            for motion_name in object_motions:
                # For each motion, store one entry for each of its limbs
                for limb_idx, limb in enumerate(kinematic_chains):
                    self.motion_clips.append(motion_name)
                    self.limb_keys.append(f"{object_type}_{limb_idx}")

    def __len__(self):
        return len(self.motion_clips)

    def __getitem__(self, item):
        motion_name = self.motion_clips[item]
        limb_key = self.limb_keys[item]
        
        object_type, limb_idx_str = limb_key.split('_')
        limb_idx = int(limb_idx_str)

        # 1. Load full-body motion and metadata
        full_motion = np.load(pjoin(self.opt.motion_dir, motion_name))
        conditions = self.cond_dict[object_type]
        
        # 2. Select the limb (kinematic chain)
        limb_chain = conditions['kinematic_chains'][limb_idx]
        
        # 3. Extract limb motion data (6D rotations are features 3 to 9)
        limb_motion_6d = full_motion[:, limb_chain, 3:9]
        
        # 4. Slice a temporal window
        motion_length = limb_motion_6d.shape[0]
        if motion_length < self.max_motion_length:
            # Pad if too short
            limb_window = np.zeros((self.max_motion_length, limb_motion_6d.shape[1], 6))
            limb_window[:motion_length, :, :] = limb_motion_6d
        else:
            # Randomly slice if too long
            start_idx = random.randint(0, motion_length - self.max_motion_length)
            limb_window = limb_motion_6d[start_idx : start_idx + self.max_motion_length]

        # 5. Pad joints and create mask
        num_limb_joints = len(limb_chain)
        padded_limb_motion = np.zeros((self.max_motion_length, self.max_limb_joints, 6))
        padded_limb_motion[:, :num_limb_joints, :] = limb_window
        
        joint_mask = np.zeros(self.max_limb_joints, dtype=np.float32)
        joint_mask[:num_limb_joints] = 1.0

        # 6. Re-index parent array for the limb
        original_parents = conditions['parents']
        limb_parents_original_indices = [original_parents[j] for j in limb_chain]
        
        local_parents = -np.ones(self.max_limb_joints, dtype=np.int64)
        for i, joint_idx in enumerate(limb_chain):
            parent_original_idx = limb_parents_original_indices[i]
            if parent_original_idx in limb_chain:
                local_parents[i] = limb_chain.index(parent_original_idx)
            else:
                local_parents[i] = -1 # Root of the limb

        # 7. Get bone lengths for the limb
        bone_lengths = np.linalg.norm(conditions['offsets'][limb_chain], axis=1)
        padded_bone_lengths = np.zeros((self.max_limb_joints, 1), dtype=np.float32)
        padded_bone_lengths[:num_limb_joints, 0] = bone_lengths

        return (
            torch.from_numpy(padded_limb_motion).float(),
            torch.from_numpy(joint_mask).float(),
            torch.from_numpy(local_parents).long(),
            torch.from_numpy(padded_bone_lengths).float(),
            motion_length
        )

class LimbDataset(data.Dataset):
    def __init__(self, split="train", num_frames=64, **kwargs):
        abs_base_path = '.'
        opt = get_opt(None) # Device is not used here
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.max_motion_length = min(opt.max_motion_length, num_frames)
        self.opt = opt
        
        print('Loading Truebones dataset for limb embedding...')
        cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
        
        if 'objects_subset' in kwargs and kwargs['objects_subset'] is not None:
            subset = opt.subsets_dict[kwargs['objects_subset']]
            cond_dict = {k: cond_dict[k] for k in subset if k in cond_dict}
            print(f"Using subset '{kwargs['objects_subset']}' with {len(cond_dict.keys())} characters.")

        self.motion_dataset = LimbMotionDataset(self.opt, cond_dict)

    def __getitem__(self, item):
        return self.motion_dataset[item]

    def __len__(self):
        return len(self.motion_dataset)

class LimbSampler(WeightedRandomSampler):
    def __init__(self, data_source):
        num_samples = len(data_source)
        
        # Create a weight for each sample (motion_clip, limb_key)
        # Goal: Balance across object types
        
        object_types = list(data_source.motion_dataset.cond_dict.keys())
        object_type_counts = {obj: 0 for obj in object_types}
        
        for limb_key in data_source.motion_dataset.limb_keys:
            obj_type = limb_key.split('_')[0]
            object_type_counts[obj_type] += 1
            
        weights = np.zeros(num_samples)
        object_share = 1.0 / len(object_types)

        for i, limb_key in enumerate(data_source.motion_dataset.limb_keys):
            obj_type = limb_key.split('_')[0]
            # The probability of this sample is the total share for its object type,
            # divided by the number of samples that object type has.
            weights[i] = object_share / object_type_counts[obj_type]
            
        super().__init__(weights=weights, num_samples=num_samples)