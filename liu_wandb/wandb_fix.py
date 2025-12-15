import wandb  # W&B SDK，用于初始化 Run 和日志记录 :contentReference[oaicite:9]{index=9}

x_minutes = [30.0, 61.34426, 95.58272, 131.54239,
            164.86831, 182.18272, 200.92387, 224.60496, 240.86420]
y_values  = [7462179.5, 7764282, 7991735, 9196473,
            8367756.5, 8432530, 10705619, 11429781, 12472172]

# 3. 初始化一个新的 W&B Run
wandb.init(
    project="MORL-baselines",  # 替换为你的项目名 :contentReference[oaicite:12]{index=12}
    entity="lx59ling-beijing-institute-of-technology"     # 替换为你的 W&B 账号或团队名 :contentReference[oaicite:13]{index=13}
)


for t, hv in zip(x_minutes, y_values):
    # 在一次 log 中同时上传横坐标和纵坐标
    wandb.log({
        "Time(minutes)"      : t,
        "eval/hypervolume"   : hv
    }) 

# 6. 结束 Run
wandb.finish()  # :contentReference[oaicite:17]{index=17}
