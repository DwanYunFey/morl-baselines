import mo_gymnasium as mo_gym
import numpy as np
import torch  #noqa: F401

from morl_baselines.multi_policy.mopac.mopac import MOPAC


def main():
    gamma = 0.99

    env = mo_gym.make("mo-hopper-2obj-v5")
    eval_env = mo_gym.make("mo-hopper-2obj-v5")

    algo = MOPAC(
        env=env,
        learn_frequency = 1,
        eval_episodes = 10,
        warmup_steps = 60000,
        ref_point = np.array([-100.0, -100.0]),
        gamma=gamma,
        log=True,
        seed=42,
    )

    algo.train(
        eval_env=eval_env,
        total_timesteps=int(6e6),
        eval_frequency=60000,
    )


if __name__ == "__main__":
    main()
