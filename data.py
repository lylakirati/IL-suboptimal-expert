import numpy as np
import pickle
import torch

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

def fetch_expert_traj_regular(size):
    obs, actions, rewards, dones, next_obs = ds.batch(batch_size=int(size), split='train')

    Expert_Flattened_States = obs[:, 0, :, :].reshape(-1, 84 * 84)
    Expert_actions = actions

    return Expert_Flattened_States, Expert_actions

def fetch_expert_traj_cnn(size):
    obs, actions, rewards, dones, next_obs = ds.batch(batch_size=int(size), split='train')

    Expert_Flattened_States = obs
    Expert_actions = actions

    return Expert_Flattened_States, Expert_actions

def fetch_expert_traj(size, args):
    if args.test == "true":
        return get_test_dataset(size)
    elif args.platform == "sklearn":
        return fetch_expert_traj_regular(size)
    elif args.platform == "nn":
        if args.nn_type == "ffn":
            return fetch_expert_traj_regular(size)
        elif args.nn_type == "cnn":
            return fetch_expert_traj_cnn(size)

def get_test_dataset(size):
    X = torch.rand(size, 6)
    y = torch.sum(X, dim = -1).type(torch.long)
    return X, y


# states, actions = get_expert_traj()
# print(states.shape)