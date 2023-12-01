from model import *
from data import fetch_expert_traj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--platform", type=str, default="nn")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--n_layer", type=int, default=3)
parser.add_argument("--state_size", type=int, default=None)
parser.add_argument("--action_size", type=int, default=None)
parser.add_argument("--data_size", type=int, default=1e4)
parser.add_argument("--train_size", type=float, default=0.7)
parser.add_argument("--val_size", type=float, default=0.1)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

if __name__ == "__main__":
    model = None
    ## TODO: fetch data
    states, actions = fetch_expert_traj(size = args.data_size)
    print("got trajectories")
    trainstates = states[:int(args.data_size * args.train_size)]
    valstates = states[int(args.data_size * args.train_size):int(args.data_size * (args.train_size + args.val_size))]
    teststates =  states[int(args.data_size * (args.train_size + args.val_size)):]

    trainactions = actions[:int(args.data_size * args.train_size)]
    valactions= actions[int(args.data_size * args.train_size):int(args.data_size * (args.train_size + args.val_size))]
    testactions =  actions[int(args.data_size * (args.train_size + args.val_size)):]

    traindata = {"states": trainstates, "actions": trainactions}
    valdata = {"states": valstates, "actions": valactions}
    testdata = {"states": teststates, "actions": testactions}
    if args.platform == "sklearn":
        ## NOTE: spacify your model if using sklearn 
        model = None
    result = run_experiment(args, traindata, testdata, valdata, model)
    print(result)

