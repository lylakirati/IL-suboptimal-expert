import numpy as np
import random
import torch
from ncps.datasets.torch import AtariCloningDataset

from eorl import OfflineDataset

ds = OfflineDataset(
    env = 'Pong',            # pass name in `supported environments` below
    dataset_size = 200000,   # [0, 1e6) frames of atari
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

def get_data_alt2(size, args):
    states = []
    actions = []
    train_ds = AtariCloningDataset("breakout", split="train")
    random_numbers = random.sample(range(0, 30000), int(args.data_size))
    for i in random_numbers:
        cur_batch_states = train_ds.__getitem__(i)[0]
        cur_batch_actions = train_ds.__getitem__(i)[1]
        if i == int(size/2):
            print("half completed")
        for j in range(0, 32):
            if args.platform == "sklearn":
                states.append((cur_batch_states.numpy()[j]).flatten())
            elif args.platform == "nn":
                states.append(cur_batch_states.numpy()[j])
            actions.append(cur_batch_actions.numpy()[j])
    if args.suboptimal == 1:
        pass
    elif args.suboptimal == 2:
        pass
    return np.array(states), np.array(actions)
#     val_ds = AtariCloningDataset("breakout", split="val")
#     test_ds = AtariCloningDataset("breakout", split="val")
    
#     train = {"states" :  torch.Tensor(size = (0, 4, 84, 84)), "actions": torch.Tensor(size = (0,))}
#     val = {"states" :  torch.Tensor(size = (0, 4, 84, 84)), "actions": torch.Tensor(size = (0,))}
#     test = {"states" :  torch.Tensor(size = (0, 4, 84, 84)), "actions": torch.Tensor(size = (0,))}

#     for i, d in enumerate(train_ds):
#         train["states"] = torch.cat((train["states"], d[0]), dim = 0)
#         train["actions"] = torch.cat((train["actions"], d[1]), dim = 0)
#         print("\rfinished {} %".format(i/len(train_ds) * 100), end = "")
#     for i, d in enumerate(val_ds):
#         val["states"] = torch.cat((val["states"], d[0]), dim = 0)
#         val["actions"] = torch.cat((val["actions"], d[1]), dim = 0)
#         print("\rfinished {} %".format(i/len(val_ds) * 100), end = "")
#     for i, d in enumerate(test_ds):
#         test["states"] = torch.cat((test["states"], d[0]), dim = 0)
#         test["actions"] = torch.cat((test["actions"], d[1]), dim = 0)
#         print("\rfinished {} %".format(i/len(test_ds) * 100), end = "")
#     return train, val, test



    
    