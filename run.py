from model import *
from data import fetch_expert_traj
from data import get_data_alt2
import argparse
import torch 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--platform", type=str, default="nn")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--n_layer", type=int, default=2)
parser.add_argument("--state_size", type=int, default=28224)
parser.add_argument("--action_size", type=int, default=18)
parser.add_argument("--data_size", type=int, default=3000)
parser.add_argument("--train_size", type=float, default=0.7)
parser.add_argument("--val_size", type=float, default=0.1)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--nn_type", type=str, default="cnn")
parser.add_argument("--test", type=str, default="false")
parser.add_argument("--alt", type=str, default="false")
parser.add_argument("--suboptimal_type", type=str)
parser.add_argument("--suboptimal_portion", type=float, default = 0.2)
parser.add_argument("--downsample_action", type=int, default = 0)
# alt = true if use CNN to train the whole 960k dataset
parser.add_argument("--suboptimal", type=int, default=0)
#

args = parser.parse_args()

if __name__ == "__main__":
    ## set device 
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else: 
        args.device = torch.device("cpu")
    
    ## for testing nn implementation 
    
    if args.test == "true":
        args.state_size = 6
        
    ## fetch data

    states, actions = get_data_alt2(size = args.data_size, args = args)
    print("got trajectories")
    trainstates = states[:int(args.data_size * args.train_size)]
    valstates = states[int(args.data_size * args.train_size):int(args.data_size * (args.train_size + args.val_size))]
    teststates = states[int(args.data_size * (args.train_size + args.val_size)):]

    trainactions = actions[:int(args.data_size * args.train_size)]
    valactions= actions[int(args.data_size * args.train_size):int(args.data_size * (args.train_size + args.val_size))]
    testactions =  actions[int(args.data_size * (args.train_size + args.val_size)):]

    traindata = {"states": trainstates, "actions": trainactions}
    valdata = {"states": valstates, "actions": valactions}
    testdata = {"states": teststates, "actions": testactions}

    print(f"train state shape: {traindata['states'].shape}")
    print(f"train action shape: {traindata['actions'].shape}")

    if args.platform == "sklearn":
        ## NOTE: spacify your model if using sklearn 
        model = DecisionTreeClassifier(max_depth = 80)
        # model = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=100)
    else:
        model = None
    result = run_experiment(args, traindata, testdata, valdata, model)
    print(result)

