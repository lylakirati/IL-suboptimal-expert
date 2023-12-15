#python run.py --platform sklearn --data_size 3000 --nn_type cnn --action_size 4 --max_depth 40
#python run.py --platform sklearn --data_size 3000 --nn_type cnn --action_size 4 --max_depth 50
#python run.py --platform sklearn --data_size 3000 --nn_type cnn --action_size 4 --max_depth 60
#python run.py --platform sklearn --data_size 300 --nn_type cnn --action_size 4 --max_depth 40
#python run.py --platform sklearn --data_size 300 --nn_type cnn --action_size 4 --max_depth 50
#python run.py --platform sklearn --data_size 300 --nn_type cnn --action_size 4 --max_depth 60

#random forest
#python run.py --platform sklearn --data_size 3000 --action_size 4 --max_depth 50 --n_estimators 100
#python run.py --platform sklearn --data_size 3000 --action_size 4 --max_depth 60 --n_estimators 100
#python run.py --platform sklearn --data_size 3000 --action_size 4 --max_depth 50 --n_estimators 200
#python run.py --platform sklearn --data_size 3000 --action_size 4 --max_depth 60 --n_estimators 200
#python run.py --platform sklearn --data_size 300 --action_size 4 --max_depth 50 --n_estimators 100
#python run.py --platform sklearn --data_size 300 --action_size 4 --max_depth 60 --n_estimators 100
#python run.py --platform sklearn --data_size 300 --action_size 4 --max_depth 50 --n_estimators 200
#python run.py --platform sklearn --data_size 300 --action_size 4 --max_depth 60 --n_estimators 200
# bash run_exp.sh
# zsh run_exp.sh

#python run.py --alt true --suboptimal 1 --suboptimal_type alter_actions --suboptimal_portion 0.5
#python run.py --alt true --suboptimal 1 --suboptimal_type alter_actions --suboptimal_portion 0.2
#python run.py --alt true --suboptimal 1 --suboptimal_type alter_actions --suboptimal_portion 0.1
#python run.py --alt true --suboptimal 1 --suboptimal_type alter_actions --suboptimal_portion 0.05
#python run.py --alt true --suboptimal 1 --suboptimal_type alter_actions --suboptimal_portion 0.01

#python run.py --alt true --suboptimal 1 --suboptimal_type downsample --suboptimal_portion 0.2
#python run.py --alt true --suboptimal 1 --suboptimal_type downsample --suboptimal_portion 0.5
#python run.py --alt true --suboptimal 1 --suboptimal_type downsample --suboptimal_portion 0.5 --downsample_action 1
#python run.py --alt true --suboptimal 1 --suboptimal_type downsample --suboptimal_portion 0.5 --downsample_action 2
#python run.py --alt true --suboptimal 1 --suboptimal_type downsample --suboptimal_portion 0.5 --downsample_action 3

