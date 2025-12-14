#!/usr/bin/env bash

# set -e：只要有任何一条命令返回非 0，整个脚本立即退出
set -e

# Darwin环境下不能跑gpi和ols算法
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    conda env create -f environment.darwin.yml || true
elif [ "$OS" = "Linux" ]; then
    conda env create -f environment.linux.yml || true
fi

conda activate morl
# 依赖安装
python -c "import morl_baselines; print('morl_baselines.__file__')"

# wandb登录
cat <<'EOF' >> ~/.netrc
machine api.wandb.ai
  login user
  password b8826bf5fd0bd6e612b9489e259f4f3669dbce35
EOF

python -c "import wandb; wandb.login()"


# 如果没有成功以开发者模式安装依赖，则
# pip uninstall morl-baselines -y 
# pip install -e .
# pip list | grep morl-baselines