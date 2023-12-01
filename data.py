import numpy as np
import pickle

from eorl import OfflineDataset

ds = OfflineDataset(
    env = 'Pong',            # pass name in `supported environments` below
    dataset_size = 40000,   # [0, 1e6) frames of atari
    train_split = 0.9,       # 90% training, 10% held out for testing
    obs_only = False,        # only get observations (no actions, rewards, dones)
    framestack = 1,          # number of frames per sample
    shuffle = True,          # chronological samples if False, randomly sampled if true
    stride = 1,               # return every stride`th chunk (where chunk size == `framestack)
    verbose = 1              # 0 = silent, >0 for reporting
)

def get_expert_traj():
    obs, actions, rewards, dones, next_obs = ds.batch(batch_size=100000, split='train')

    Expert_Flattened_States = obs[:, 0, :, :].reshape(-1, 84 * 84)
    Expert_actions = actions

    return Expert_Flattened_States, Expert_actions

states, actions = get_expert_traj()
print(states.shape)