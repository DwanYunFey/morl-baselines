conda env create -f environment.yml
conda activate morl
# code online.ipynb
echo "open the online.ipynb and click the 'Run' button"

python ./morl_baselines/common/check_pip_editable.py
status=$?
if [ $status -ne 0 ]; then
    pip uninstall morl-baselines -y
    pip install -e .
    pip list | grep morl-baselines
fi

python ./morl_baselines/common/device.py
# 访问 https://wandb.ai/authorize 获取 passwords
# wandb登录
cat <<'EOF' >> ~/.netrc
machine api.wandb.ai
  login user
  password $(wandb api key)
EOF

python -c "import wandb; wandb.login()"