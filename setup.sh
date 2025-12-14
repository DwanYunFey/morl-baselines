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

python check_morl_editable.py
status=$?
if [ $status -ne 0 ]; then
    pip uninstall morl-baselines -y
    pip install -e .
    pip list | grep morl-baselines
fi


# wandb登录
cat <<'EOF' >> ~/.netrc
machine api.wandb.ai
  login user
  password b8826bf5fd0bd6e612b9489e259f4f3669dbce35
EOF

python -c "import wandb; wandb.login()"

echo "open the online.ipynb and click the 'Run' button"