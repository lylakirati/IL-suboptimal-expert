import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
# import evaluate 
from sklearn.metrics import f1_score, accuracy_score
from ncps.datasets.torch import AtariCloningDataset
import numpy as np

import gym
import ale_py
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from torch.utils.data import Dataset
import torch.optim as optim


from ncps.torch import CfC
from ncps.datasets.torch import AtariCloningDataset


class nn_agent(nn.Module):
    def __init__(self, n_layer, state_size, action_size):
        super().__init__()
        self.state_size =  state_size
        self.action_size = action_size
        self.n_layer = n_layer
        layers = nn.Sequential(nn.Linear(self.state_size, self.state_size), nn.ReLU())
        for _ in range(self.n_layer):
            layers.append(nn.Linear(self.state_size, self.state_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.state_size, self.action_size))
        self.layers = layers
        
    
    def forward(self, data):
        return self.layers(data)

class cnn_agent(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        # self.conv3 = nn.Conv2d(128, 256, 3)
        # self.conv4 = nn.Conv2d(256, 256, 3)
        # 1 layer: 107584
        # 2 layers: 46208
        # 3 layers: 16384
        # 4 layers: 2304
        self.fc1 = nn.Linear(46208, 120) # 144 5776
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class nn_dataset(Dataset):
    def __init__(self, data):
        self.X = data["states"]
        self.y = data["actions"]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            "states": self.X[idx],
            "actions": self.y[idx]
        }
    
class nn_bc_classifier:
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = args.lr)
        self.criterion = nn.CrossEntropyLoss()

    
    def train(self, trainloader, valloader, testloader):
        val_metrics_log = {"f1": [], "accuracy": []}
        test_metrics_log = {"f1": [], "accuracy": []}
        for i in range(self.args.n_epochs):
            tbar = tqdm(trainloader, dynamic_ncols=True)
            self.model.train()
            for batch in tbar:
                batch = self._process_batch(batch)
                self.optimizer.zero_grad()
                logits = self.model(torch.tensor(batch["states"], dtype = torch.float).to(self.args.device))
                loss = self.criterion(logits, torch.tensor(batch["actions"], dtype = torch.long).to(self.args.device))
                loss.backward()
                self.optimizer.step()
                # print(loss.item())
                tbar.set_postfix(loss = loss.item())
            val_metrics = self._eval(valloader)
            test_metrics = self._eval(testloader)
            val_metrics_log["f1"].append(val_metrics["f1"])
            val_metrics_log["accuracy"].append(val_metrics["accuracy"])
            test_metrics_log["f1"].append(test_metrics["f1"])
            test_metrics_log["accuracy"].append(test_metrics["accuracy"])
            print("=" * 20)
            test_acc = test_metrics["accuracy"]
            val_acc = val_metrics["accuracy"]
            print(f"test accuracy is {test_acc }")
            print(f"val accuracy is {val_acc }")
            # print(print(self.model.layers[0].bias))
            print("=" * 20)
        return val_metrics_log, test_metrics_log
            
        
    def _eval(self, dataloader):
        preds = []
        actions = []
        self.model.eval()
        tbar = tqdm(dataloader, dynamic_ncols=True)
        with torch.no_grad():
            for batch in tbar:
                batch = self._process_batch(batch)
                logits = self.model(torch.tensor(batch["states"], dtype = torch.float).to(self.args.device))
                pred = torch.argmax(logits, dim = -1)
                preds.extend(pred.detach().cpu().tolist())
                # print(pred.tolist())
                # print(batch["actions"].tolist())
                actions.extend(batch["actions"].tolist())
        # print(actions)
        # print(preds)
        f1 = f1_score(y_true= actions, y_pred=preds, average = "macro")
        accuracy = accuracy_score(y_true= actions, y_pred=preds)
        return {
            "f1": f1,
            "accuracy": accuracy
        }
    
    def _process_batch(self, batch):
        if self.args.alt == "true":
            batch = {"states": torch.squeeze(batch[0]), "actions" : torch.squeeze(batch[1])}
            if self.args.nn_type == "ffn":
                batch["states"] = torch.flatten(batch["states"], start_dim = 1, end_dim = -1)
            if self.args.suboptimal == 1:
                batch = perturb(self.args, batch)
        return batch
     
    def experiment(self, trainloader, valloader, testloader):
        val_metrics_log, test_metrics_log = self.train(trainloader, valloader, testloader)
        best_f1_index = np.argmax(val_metrics_log["f1"]).item()
        best_acc_index = np.argmax(val_metrics_log["accuracy"]).item()
        test_f1 = test_metrics_log["f1"][best_f1_index]
        test_acc = test_metrics_log["accuracy"][best_acc_index]
        return {
            "f1": test_f1 ,
            "accuracy": test_acc
        }

    

class sklearn_classifier:
    def __init__(self, model, args):
        self.args = args
        self.model = model
    
    def train(self, traindata):
        self.model.fit(traindata["states"], traindata["actions"])


    def experiment(self, traindata, testdata):
        self.train(traindata)
        preds = self.model.predict(testdata["states"])
        f1 = f1_score(y_true = testdata["actions"], y_pred=preds, average = "macro")
        accuracy = accuracy_score(y_true= testdata["actions"], y_pred=preds)
        return {
            "f1": f1,
            "accuracy": accuracy
        }

def run_experiment(args, traindata, testdata, valdata = None, model = None):
    if args.platform == "nn":
        return run_experiment_nn(args, traindata, valdata, testdata)
    if args.platform == "sklearn":
        return run_experiment_sklearn(args, model, traindata, testdata)


def run_experiment_sklearn(args, model, traindata, testdata):
    classifier = sklearn_classifier(model, args)
    print(f"train state shape: {traindata['states'].shape}")
    result = classifier.experiment(traindata, testdata)
    # Visualize game playing
    cumulative_rewards = visualize_game(args, classifier.model)
    return result

def run_experiment_nn(args, traindata = None, valdata = None, testdata = None):
    ## load model
    if args.nn_type == "ffn":
        model = nn_agent(n_layer = args.n_layer, state_size = args.state_size , action_size = args.action_size )
    elif args.nn_type == "cnn":
        model = cnn_agent(action_size = args.action_size)
    model.to(args.device)

    ## load classifier
    classifier = nn_bc_classifier(model, args)
    ## load dataset 
    traindataset = nn_dataset(traindata)
    valdataset = nn_dataset(valdata)
    testdataset = nn_dataset(testdata)

    # ## create data loader
    trainloader = DataLoader(traindataset, batch_size = args.bs, shuffle = False)
    valloader = DataLoader(valdataset, batch_size = args.bs, shuffle = False)
    testloader = DataLoader(testdataset, batch_size = args.bs, shuffle = False)
    if args.alt == "true":
        trainloader, valloader, testloader = get_data_alt()

    ## get results
    result = classifier.experiment(trainloader, valloader, testloader)

    # Visualize game playing
    cumulative_rewards = visualize_game(args, classifier.model)

    return result

def get_data_alt():
    train_ds = AtariCloningDataset("breakout", split="train")
    val_ds = AtariCloningDataset("breakout", split="val")
    test_ds = AtariCloningDataset("breakout", split="val")
    trainloader = DataLoader(train_ds, batch_size = 1, shuffle=True)
    valloader = DataLoader(val_ds, batch_size = 1, shuffle=True)
    testloader = DataLoader(test_ds, batch_size = 1, shuffle=True)
    return trainloader, valloader, testloader


def run_closed_loop(args, model, env, num_episodes=10):
    obs = env.reset()
    print(f"obs shape: {obs.shape}")
    # device = next(model.parameters()).device
    returns = []
    total_reward = 0

    # sklearn
    if args.platform == "sklearn":
        while True:
            obs = np.transpose(obs, [2, 0, 1]).astype(np.float32)
            pred = model.predict(obs.reshape(1, -1))

            obs, r, done, _ = env.step(pred[0])
            total_reward += r
            if done:
                obs = env.reset()
                # hx = None  # Reset hidden state of the RNN
                returns.append(total_reward)
                total_reward = 0
                if num_episodes is not None:
                    # Count down the number of episodes
                    num_episodes = num_episodes - 1
                    if num_episodes == 0:
                        print(returns)
                        return returns # returns cumulative rewards

    elif args.platform == "nn":
        with torch.no_grad():
            while True:
                # PyTorch require channel first images -> transpose data
                obs = np.transpose(obs, [2, 0, 1]).astype(np.float32)
                # add batch and time dimension (with a single element in each)
                obs = torch.from_numpy(obs).unsqueeze(0).to(args.device)
                pred = model(obs)
                action = torch.argmax(pred, dim = -1).item()
                # remove time and batch dimension -> then argmax
                # action = pred.argmax().item()
                obs, r, done, _ = env.step(action)
                total_reward += r
                if done:
                    obs = env.reset()
                    hx = None  # Reset hidden state of the RNN
                    returns.append(total_reward)
                    total_reward = 0
                    if num_episodes is not None:
                        # Count down the number of episodes
                        num_episodes = num_episodes - 1
                        if num_episodes == 0:
                            return returns # returns cumulative rewards


def visualize_game(arg, model):
    # # Visualize Atari game and play endlessly
    # ale = ALEInterface()
    # ale.loadROM(Breakout)
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = wrap_deepmind(env)
    cumulative_rewards = run_closed_loop(arg, model, env)
    print(f"Mean return {np.mean(cumulative_rewards)} (n={len(cumulative_rewards)})")
    return cumulative_rewards

def perturb(args, traindata):
    if args.suboptimal_type == "alter_actions":
        # Alter expert actions in train to be incorrect A\{a} for the speficied portion
        # print("Suboptimal type: alter incorrect expert actions")
        # print(f"Alter portion: {args.suboptimal_portion * 100: .1f}%")
        train_size = traindata['actions'].shape[0]
        alter_idx = np.random.choice(train_size, int(args.suboptimal_portion * train_size), replace = False)
        action_set = np.array(range(4)) # array of all possible actions (0, 1, 2 , 3)
        for i in alter_idx:
            # select an action randomly from A\{a} where a is the true expert action
            traindata['actions'][i] = np.random.choice(np.delete(action_set, traindata['actions'][i]))
    elif args.suboptimal_type == "downsample":
        # downsample expert action specified in downsample_action
        # print("Suboptimal type: downsample a certain action")
        # print(f"Downsampled action: {args.downsample_action}")
        # print(f"Alter portion (as a % of the size of the original action): {args.suboptimal_portion * 100: .1f}%")
        action_size = (traindata['actions'] == args.downsample_action).sum()
        alter_portion = int(args.suboptimal_portion * action_size)
        certain_action_idx = np.where(traindata['actions'] == args.downsample_action)[0]
        # sample observations to delete (downsample)
        to_delete_idx = np.random.choice(certain_action_idx, alter_portion, replace = False)
        # delete from train dataset
        traindata['states'] = np.delete(traindata['states'], to_delete_idx, 0)
        traindata['actions'] = np.delete(traindata['actions'], to_delete_idx)
    elif args.suboptimal_type == "limit_size":
        # randomly select observations for suboptimal_portion * traindata_size to delete
        # this will shrink the expert data size by suboptimal_portion
        print("Suboptimal type: limit expert data size")
        print(f"Reduce expert size by: {args.suboptimal_portion * 100: .1f}%")
        train_size = traindata['actions'].shape[0]
        to_delete_idx = np.random.choice(train_size, int(args.suboptimal_portion * train_size), replace = False)
        # delete from train dataset
        traindata['states'] = np.delete(traindata['states'], to_delete_idx, 0)
        traindata['actions'] = np.delete(traindata['actions'], to_delete_idx)
    return traindata
