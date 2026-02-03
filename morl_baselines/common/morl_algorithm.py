"""MORL algorithm base classes."""

import os
from datetime import datetime
from abc import ABC, abstractmethod
from distutils.util import strtobool
from typing import Dict, Literal, Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn
import wandb
from gymnasium import spaces
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv  # 多目标环境向量化包装
from morl_baselines.common.device import avilable_device  # 自动检测可用设备
from morl_baselines.common.evaluation import (
    eval_mo_reward_conditioned,  # 单步多目标策略评估
    policy_evaluation_mo,  # 多集多目标策略评估
)


class MOPolicy(ABC):
    """An MORL policy.

    It has an underlying learning structure which can be:
    - used to get a greedy action via eval()
    - updated using some experiences via update()

    Note that the learning structure can embed multiple policies (for example using a Conditioned Network).
    In this case, eval() requires a weight vector as input.
    """

    def __init__(self, id: Optional[int] = None, device: Union[th.device, str] = "auto") -> None:
        """Initializes the policy.

        Args:
            id: The id of the policy
            device: The device to use for the tensors
        """
        self.id = id  # 策略ID（区分多策略场景中的不同策略）
        self.device = avilable_device() if device == "auto" else device  # 计算设备
        self.global_step = 0  # 全局步数计数器

    @abstractmethod
    def eval(self, obs: np.ndarray, w: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        """Gives the best action for the given observation.

        Args:
            obs (np.array): Observation
            w (optional np.array): weight for scalarization

        Returns:
            np.array or int: Action
        """

    def __report(
        self,
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        discounted_vec_return,
    ):
        """Writes the data to wandb summary."""
        # 根据策略ID生成标记字符串
        if self.id is None:
            idstr = ""
        else:
            idstr = f"_{self.id}"

        # 记录加权标量回报
        wandb.log(
            {
                f"eval{idstr}/scalarized_return": scalarized_return,  # 平均加权回报
                f"eval{idstr}/scalarized_discounted_return": scalarized_discounted_return,  # 折扣加权回报
                "global_step": self.global_step,
            }
        )
        # 记录每个目标维度的回报
        for i in range(vec_return.shape[0]):
            wandb.log(
                {
                    f"eval{idstr}/vec_{i}": vec_return[i],  # 第i个目标的平均回报
                    f"eval{idstr}/discounted_vec_{i}": discounted_vec_return[i],  # 折扣回报
                },
            )

    def policy_eval(
        self,
        eval_env,
        num_episodes: int = 5,
        scalarization=np.dot,
        weights: Optional[np.ndarray] = None,
        log: bool = False,
    ):
        """Runs a policy evaluation (typically over a few episodes) on eval_env and logs some metrics if asked.

        Args:
            eval_env: evaluation environment
            num_episodes: number of episodes to evaluate
            scalarization: scalarization function
            weights: weights to use in the evaluation
            log: whether to log the results

        Returns:
             a tuple containing the average evaluations
        """
        (
            scalarized_return,
            scalarized_discounted_return,
            vec_return,
            discounted_vec_return,
        ) = policy_evaluation_mo(self, eval_env, scalarization=scalarization, w=weights, rep=num_episodes)  # 评估策略性能

        # 如果需要，记录评估结果到Wandb
        if log:
            self.__report(
                scalarized_return,
                scalarized_discounted_return,
                vec_return,
                discounted_vec_return,
            )

        return (
            scalarized_return,
            scalarized_discounted_return,
            vec_return,
            discounted_vec_return,
        )

    def policy_eval_esr(
        self,
        eval_env,
        scalarization,
        weights: Optional[np.ndarray] = None,
        log: bool = False,
    ):
        """Runs a policy evaluation (typically on one episode) on eval_env and logs some metrics if asked.

        Args:
            eval_env: evaluation environment
            scalarization: scalarization function
            weights: weights to use in the evaluation
            log: whether to log the results

        Returns:
             a tuple containing the evaluations
        """
        (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward,
        ) = eval_mo_reward_conditioned(self, eval_env, scalarization, weights)

        if log:
            self.__report(
                scalarized_reward,
                scalarized_discounted_reward,
                vec_reward,
                discounted_vec_reward,
            )

        return (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward,
        )

    def get_policy_net(self) -> torch.nn.Module:
        """Returns the weights of the policy net."""
        pass

    def get_buffer(self):
        """Returns a pointer to the replay buffer."""
        pass

    def set_buffer(self, buffer):
        """Sets the buffer to the passed buffer.

        Args:
            buffer: new buffer (potentially shared)
        """
        pass

    def get_save_dict(self, save_replay_buffer: bool = False) -> dict:
        """Returns a dictionary of the policy's weights and replay buffer.

        Args:
            save_replay_buffer: whether to save the replay buffer

        Returns:
            dict: dictionary of the policy's weights and replay buffer
        """
        pass

    def save(
        self,
        save_dir: str = "weights/",
        filename: Optional[str] = None,
        save_replay_buffer: bool = False,
    ):
        """Save the agent's weights and replay buffer."""
        os.makedirs(save_dir, exist_ok=True)
        filename = filename or f"policy_{self.id}.pth"
        save_path = os.path.join(save_dir, filename)
        save_dict = self.get_save_dict(save_replay_buffer)
        th.save(save_dict, save_path)

    def load(self, path, load_replay_buffer=True):
        """Load the agent's weights and replay buffer."""
        pass

    def set_weights(self, weights: np.ndarray):
        """Sets new weights.

        Args:
            weights: the new weights to use in scalarization.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """Update algorithm's parameters (e.g. using experiences from the buffer)."""


class MOAgent(ABC):
    """An MORL Agent, can contain one or multiple MOPolicies. Contains helpers to extract features from the environment, setup logging etc."""

    def __init__(
        self,
        env: Optional[gym.Env],
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the agent.

        Args:
            env: (gym.Env): The environment
            device: (str): The device to use for training. Can be "auto", "cpu" or "cuda".
            seed: (int): The seed to use for the random number generator
        """
        self.extract_env_info(env)  # 提取环境信息（观察空间、动作空间等）
        self.device = avilable_device() if device == "auto" else device  # 设置计算设备
        if isinstance(self.device, str):
            self.device = th.device(self.device)
        self.global_step = 0  # 全局时间步计数器
        self.num_episodes = 0  # ep计数器
        self.seed = seed  # 随机种子
        self.np_random = np.random.default_rng(self.seed)  # NumPy随机数生成器

    def extract_env_info(self, env: Optional[gym.Env]) -> None:
        """Extracts all the features of the environment: observation space, action space, ...

        Args:
            env (gym.Env): The environment
        """
        # Sometimes, the environment is not instantiated at the moment the MORL algorithms is being instantiated.
        # So env can be None. It is the responsibility of the implemented MORLAlgorithm to call this method in those cases
        # 环境可能在初始化时还未实例化，因此可能为None
        if env is not None:
            self.env = env
            # 处理离散观察空间
            if isinstance(self.env.observation_space, spaces.Discrete):
                self.observation_shape = (1,)
                self.observation_dim = self.env.observation_space.n  # 离散观察空间的大小
            else:
                # 处理连续观察空间
                self.observation_shape = self.env.observation_space.shape
                self.observation_dim = self.env.observation_space.shape[0]  # 观察维度

            self.action_space = env.action_space
            # 处理离散动作空间
            if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiBinary)):
                self.action_shape = (1,)
                self.action_dim = self.env.action_space.n  # 离散动作空间大小
            else:
                # 处理连续动作空间
                self.action_shape = self.env.action_space.shape
                self.action_dim = self.env.action_space.shape[0]  # 动作维度

            # 多目标奖励维度
            self.reward_dim = self.env.unwrapped.reward_space.shape[0]

    @abstractmethod
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration.

        Returns:
            dict: Config
        """

    def register_additional_config(self, conf: Dict = {}) -> None:
        """Registers additional config parameters to wandb. For example when calling train().

        Args:
            conf: dictionary of additional config parameters
        """
        # 将额外的配置参数添加到Wandb
        for key, value in conf.items():
            wandb.config[key] = value

    def setup_wandb(
        self,
        project_name: str,
        experiment_name: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        mode: Literal["online", "offline", "disabled", "shared"] | None = "offline",
    ) -> None:
        """Initializes the wandb writer.

        Args:
            project_name: name of the wandb project. Usually MORL-Baselines.
            experiment_name: name of the wandb experiment. Usually the algorithm name.
            entity: wandb entity. Usually your username but useful for reporting other places such as openrlbenmark.

        Returns:
            None
        """
        self.experiment_name = experiment_name  # 实验名称
        # 获取环境ID（处理向量化环境情况）
        env_id = self.env.spec.id if not isinstance(self.env, MOSyncVectorEnv) else self.env.envs[0].spec.id
        cur = datetime.now().strftime("%Y%m%d%H%M%S")  # 当前时间戳
        # 生成唯一的实验名称：环境_算法_种子_时间
        self.full_experiment_name = f"{env_id}_{experiment_name}_{self.seed}_{cur}"
        import wandb

        # 获取算法配置并添加额外信息
        config = self.get_config()
        config["algo"] = self.experiment_name
        config["device"] = str(self.device)
        # 检查是否监控Gymnasium环境
        monitor_gym = strtobool(os.environ.get("MONITOR_GYM", "True"))

        # 设置Wandb实体（默认为特定团队）
        if entity is None:
            team = "duan-yun-fei-beijing-institute-of-technology"
            entity = os.environ.get("WANDB_ENTITY", team)
        
        # 设置Wandb模式（在线/离线/禁用）
        if mode is None:
            mode = os.environ.get("WANDB_MODE", "offline")
        
        wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            name=self.full_experiment_name,
            monitor_gym=monitor_gym,
            save_code=True,
            group=group,
            mode=mode,
        )
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self) -> None:
        """Closes the wandb writer and finishes the run."""
        import wandb

        wandb.finish()  # 完成并关闭Wandb运行
