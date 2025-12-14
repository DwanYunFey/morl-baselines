#!/bin/bash

# Darwin环境下不能跑gpi和ols算法
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    conda env create -f environment.darwin.yml
elif [ "$OS" = "Linux" ]; then
    conda env create -f environment.linux.yml
fi

python -c "import morl_baselines; print('✅ morl-baselines 依赖安装成功')"

wandb login --relogin
# wandb API Key:
# b8826bf5fd0bd6e612b9489e259f4f3669dbce35
