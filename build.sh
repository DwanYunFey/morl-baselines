#!/bin/bash
# conda环境
echo "执行conda create -n morl python=3.10"
conda create -n morl python=3.10

echo "执行conda activate morl"
conda activate morl

# Darwin环境下不能跑gpi和ols算法
# 安装依赖
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    echo "执行: pip install \".[testing]\""
    pip install ".[testing]"
elif [ "$OS" = "Linux" ]; then
    echo "执行: pip install -e \".[testing, all]\""
    pip install -e ".[testing, all]"
fi
# 检查安装是否成功
if [ $? -eq 0 ]; then
    echo "依赖安装成功！"
else
    echo "依赖安装失败！"
fi
python -c "import morl_baselines; print('✅ morl-baselines 导入成功')"

wandb login --relogin
# wandb API Key:
# b8826bf5fd0bd6e612b9489e259f4f3669dbce35
