# IL-suboptimal-expert

We provide examples on shell command to run our experiment. For example, to run the CNN experiment, do: 

- `python run.py --platform nn --nn_type cnn --alt true`

The `--platform nn` flag tells the script that we are running neural network using PyTorch. the `--nn_type cnn` command tells the script that we are using CNN (using `--nn_type ffn` tells the script to use feaddforwad neural network, which triggers memory error. To use that, you also need to specify the `--state_size` to be 84 * 84 * 4). The `--alt true` command tells the script to use all 960K data points, 

To use part of the data, use command like the following:

- `python run.py --platform nn --nn_type cnn --alt false --data_size 1000`

To use SKlearn, use command like the following, and you need to manually define the sklearn model in run.py

-  `python run.py --platform sklearn --data_size 1000`

To introduce suboptimality, use command like the following. `--suboptimal_type alter_actions` tells the script that the suboptimality is random perturbation. `--suboptimal_portion` tells that the portion to perturb is 20%. 

- `python run.py --platform nn --nn_type cnn --alt true --suboptimal_type alter_actions --suboptimal_portion 0.2`

If the suboptimality is downsample, then do something like the following, where `--downsample_action 0` tells the model to downsample action 0

- `python run.py --platform nn --nn_type cnn --alt true --suboptimal_type downsample --suboptimal_portion 0.2 --downsample_action 0`

The other arguments in the `run.py` are mostly model hyperparameters, and they should be self-explanatory (e.g., `bs` stands for batch size). You typically do not need to modify them. 
