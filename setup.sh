conda env create -f environment.yml
conda activate morl
echo "open the online.ipynb and click the 'Run' button"

python ./morl_baselines/common/check_pip_editable.py
status=$?
if [ $status -ne 0 ]; then
    pip uninstall morl-baselines -y
    pip install -e .
    pip list | grep morl-baselines
fi

python ./morl_baselines/common/device.py
# wandb登录
cat <<'EOF' >> ~/.netrc
machine api.wandb.ai
  login user
  password b8826bf5fd0bd6e612b9489e259f4f3669dbce35
EOF

python -c "import wandb; wandb.login()"