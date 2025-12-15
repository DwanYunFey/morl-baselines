# multi_run_convert.py
import os
import wandb
import pandas as pd

# —— 配置区域 —— #
entity       = "lx59ling-beijing-institute-of-technology"           # 你的 W&B 账号或团队名
src_project  = "MORL-baselines"      # 源项目名
run_ids      = ["gqj3lbga"]
dst_project  = "MORL-baselines"      # 新 Run 要存放的项目
# —— end 配置 —— #


# （可选）增加 HTTP 超时
os.environ["WANDB_HTTP_TIMEOUT"] = "120"

# 增加 GraphQL timeout 到 60 秒
api = wandb.Api(timeout=60)

for rid in run_ids:
    # 1. 拉取单个源 Run 历史
    src = api.run(f"{entity}/{src_project}/{rid}")
    df  = src.history(samples=src.lastHistoryStep + 1)  # Run.history 拉取所有记录 :contentReference[oaicite:0]{index=0}

    # 2. 只保留我们关心的列，并转换时间：
    #    W&B 系统 “Relative Time (Process)” 单位为秒，转为分钟
    df = df[["_runtime", "eval/hypervolume", "eval/eum"]]
    df = df.rename(columns={"_runtime": "Relative Time (Process)"})
    df["Time(minutes)"] = df["Relative Time (Process)"] / 60.0  # 列表/向量运算 :contentReference[oaicite:1]{index=1}

    # 3. 为每个源 Run 初始化一个新的 Run
    new_run = wandb.init(
        project=dst_project,
        entity=entity,
        name=f"{rid}_converted"
    )

    # 4. 定义自定义 X‑轴，并绑定两个指标
    #    这样 Log 时无需表格，UI 就能直接画折线图 :contentReference[oaicite:2]{index=2}
    wandb.define_metric("Time(minutes)")
    wandb.define_metric("eval/hypervolume", step_metric="Time(minutes)", step_sync=True)
    wandb.define_metric("eval/eum",        step_metric="Time(minutes)", step_sync=True)

    # 5. 逐行 log：一次调用包含横坐标、两个纵坐标，以及 commit=True 完成该步
    for row in df.to_dict(orient="records"):
        new_run.log({
            "Time(minutes)"      : row["Time(minutes)"],
            "eval/hypervolume"   : row["eval/hypervolume"],
            "eval/eum"           : row["eval/eum"],
        })  # run.log 文档 :contentReference[oaicite:3]{index=3}

    new_run.finish()
    print(f"Finished new run for source {rid}")


