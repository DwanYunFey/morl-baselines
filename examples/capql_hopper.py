import mo_gymnasium as mo_gym
import numpy as np
import torch  # noqa: F401

from morl_baselines.multi_policy.capql.capql import CAPQL


def main():
    gamma = 0.99

    env = mo_gym.make("mo-hopper-v4")
    eval_env = mo_gym.make("mo-hopper-v4")

    algo = CAPQL(
        env=env,
        gamma=gamma,
        log=True,
        gradient_updates=1,
    )

    algo.train(
        eval_env=eval_env,
        total_timesteps=int(4e6) + 1,
        ref_point=np.array([-100.0, -100.0, -100.0]),
        known_pareto_front=None,
    )


if __name__ == "__main__":
    main()
