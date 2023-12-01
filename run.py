from model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--platform", type=str, default="nn")
parser.add_argument("--bs", type=float, default=1e-2)
parser.add_argument("--n_layer", type=int, default=3)
parser.add_argument("--state_size", type=int, default=None)
parser.add_argument("--action_size", type=int, default=None)

args = parser.parse_args()

if __name__ == "__main__":
    model = None
    ## TODO: fetch data
    traindata, testdata, valdata = None, None, None
    if args.platform == "sklearn":
        ## NOTE: spacify your model if using sklearn 
        model = None
    run_experiment(args, traindata, testdata, valdata, model)

