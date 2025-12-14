# Darwin
tmux new -s torch_train
caffeinate -i python train.py --config config.yaml

# Linux
tmux new -s torch8h
python train.py --config config.yaml