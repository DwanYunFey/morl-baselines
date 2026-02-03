"""CAPQL algorithm."""

import os
import random
from itertools import chain
from typing import List, Optional, Union

import gymnasium
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Normal

from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,  # 记录多目标强化学习的关键指标（超体积、期望效用等）
    log_episode_info,
    policy_evaluation_mo,  # 评估多目标策略在给定权向量下的性能
)
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy  # 多目标强化学习的基类
from morl_baselines.common.networks import layer_init, mlp, polyak_update  # 神经网络构建和初始化工具
from morl_baselines.common.weights import equally_spaced_weights  # 生成均匀分布的权向量


LOG_SIG_MAX = 2  # 策略网络日志标准差的上界
LOG_SIG_MIN = -20  # 策略网络日志标准差的下界
EPSILON = 1e-6  # 数值稳定性常数


class ReplayMemory:
    """Replay memory."""

    def __init__(self, capacity: int):
        """Initialize the replay memory."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, weights, reward, next_state, done):
        """Push a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # 将转移(s,a,w,r,s',done)存储到缓存（以副本形式保存）
        self.buffer[self.position] = (
            np.array(state).copy(),
            np.array(action).copy(),
            np.array(weights).copy(),  # 权向量：多目标中用于条件化策略的目标权重
            np.array(reward).copy(),  # 多目标奖励向量
            np.array(next_state).copy(),
            np.array(done).copy(),
        )
        # 循环覆盖缓存（当缓存满后，从头开始覆盖最旧的转移）
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, to_tensor=True, device=None):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)  # 随机采样batch_size条经验
        # 将采样数据按维度堆叠（转换为批次形式）
        state, action, w, reward, next_state, done = map(np.stack, zip(*batch))
        experience_tuples = (state, action, w, reward, next_state, done)
        # 转换为PyTorch张量（便于GPU计算）
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x, dtype=th.float32).to(device), experience_tuples))
        return state, action, w, reward, next_state, done

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class WeightSamplerAngle:
    """Sample weight vectors from normal distribution."""

    def __init__(self, rwd_dim, angle, w=None):
        """Initialize the weight sampler."""
        self.rwd_dim = rwd_dim
        self.angle = angle
        if w is None:
            w = th.ones(rwd_dim)
        w = w / th.norm(w)
        self.w = w

    def sample(self, n_sample):
        """Sample n_sample weight vectors from normal distribution."""
        s = th.normal(th.zeros(n_sample, self.rwd_dim))  # 从高斯分布采样

        # 移除s在参考方向w上的投影（保证新采样向量与参考方向正交）
        s = s - (s @ self.w).view(-1, 1) * self.w.view(1, -1)

        # L2 归一化s向量
        s = s / th.norm(s, dim=1, keepdim=True)

        # 从[0, angle]范围内均匀采样角度
        s_angle = th.rand(n_sample, 1) * self.angle

        # 通过旋转s和参考向量w的线性组合来构造权向量
        w_sample = th.tan(s_angle) * s + self.w.view(1, -1)

        # L1 归一化权向量（确保权重求和为1）
        w_sample = w_sample / th.norm(w_sample, dim=1, keepdim=True, p=1)

        return w_sample.float()


class Policy(nn.Module):
    """Policy network."""

    def __init__(self, obs_dim, rew_dim, output_dim, action_space, net_arch=[256, 256]):
        """Initialize the policy network."""
        super().__init__()
        self.action_space = action_space
        # 输入：观察 + 权向量，输出：隐层特征（网络架构由net_arch指定）
        self.latent_pi = mlp(obs_dim + rew_dim, -1, net_arch)
        # 输出均值（用于确定性动作和高斯分布的均值）
        self.mean = nn.Linear(net_arch[-1], output_dim)
        # 输出对数标准差（log(std)，确保std > 0）
        self.log_std_linear = nn.Linear(net_arch[-1], output_dim)

        # 动作缩放参数：将tanh输出[-1,1]缩放到实际动作范围[low, high]
        self.register_buffer("action_scale", th.tensor((action_space.high - action_space.low) / 2.0, dtype=th.float32))
        self.register_buffer("action_bias", th.tensor((action_space.high + action_space.low) / 2.0, dtype=th.float32))

        self.apply(layer_init)  # 使用正交初始化所有权重

    def forward(self, obs, w):
        """Forward pass of the policy network."""
        # 拼接观察和权向量作为网络输入
        h = self.latent_pi(th.concat((obs, w), dim=obs.dim() - 1))
        mean = self.mean(h)
        log_std = self.log_std_linear(h)
        # 将log_std限制在合理范围，保证数值稳定性
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def get_action(self, obs, w):
        """Get an action from the policy network."""
        mean, _ = self.forward(obs, w)
        # 确定性动作：均值经过tanh和动作缩放（用于评估）
        return th.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, obs, w):
        """Sample an action from the policy network."""
        # 获取均值和标准差
        mean, log_std = self.forward(obs, w)
        std = log_std.exp()

        # 从高斯分布采样（重参数化技巧）
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样：mean + std * N(0,1)

        # 通过tanh限制输出到[-1, 1]范围
        y_t = th.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # 计算采样动作的对数概率
        log_prob = normal.log_prob(x_t).sum(dim=1)

        # 修正tanh变换引入的雅可比行列式
        # d(tanh(x))/dx = 1 - tanh²(x)
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON).sum(dim=1)
        log_prob = log_prob.clamp(-1e3, 1e3)  # 防止极端值

        # 确定性均值动作（用于策略梯度更新）
        mean = th.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


class QNetwork(nn.Module):
    """Q-network S x Ax W -> R^reward_dim."""

    def __init__(self, obs_dim, action_dim, rew_dim, net_arch=[256, 256]):
        """Initialize the Q-network."""
        super().__init__()
        # 输入：观察 + 动作 + 权向量，输出：多目标Q值（rew_dim维）
        self.net = mlp(obs_dim + action_dim + rew_dim, rew_dim, net_arch)
        self.apply(layer_init)

    def forward(self, obs, action, w):
        """Forward pass of the Q-network."""
        # 拼接观察、动作和权向量，计算多目标Q值
        q_values = self.net(th.cat((obs, action, w), dim=obs.dim() - 1))
        return q_values


class CAPQL(MOAgent, MOPolicy):
    """CAPQL algorithm.

    MULTI-OBJECTIVE REINFORCEMENT LEARNING: CONVEXITY, STATIONARITY AND PARETO OPTIMALITY
    Haoye Lu, Daniel Herman & Yaoliang Yu
    ICLR 2023
    Paper: https://openreview.net/pdf?id=TjEzIsyEsQ6
    Code based on: https://github.com/haoyelu/CAPQL
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1000000,
        net_arch: List = [256, 256],
        batch_size: int = 128,
        num_q_nets: int = 2,
        alpha: float = 0.2,
        learning_starts: int = 1000,
        gradient_updates: int = 1,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "CAPQL",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        """CAPQL algorithm with continuous actions.

        It extends the Soft-Actor Critic algorithm to multi-objective RL.
        It learns the policy and Q-networks conditioned on the weight vector.

        Args:
            env (gym.Env): The environment to train on.
            learning_rate (float, optional): The learning rate. Defaults to 3e-4.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The soft update coefficient. Defaults to 0.005.
            buffer_size (int, optional): The size of the replay buffer. Defaults to int(1e6).
            net_arch (List, optional): The network architecture for the policy and Q-networks. Defaults to [256, 256].
            batch_size (int, optional): The batch size for training. Defaults to 128.
            num_q_nets (int, optional): The number of Q-networks to use. Defaults to 2.
            alpha (float, optional): The entropy regularization coefficient. Defaults to 0.2.
            learning_starts (int, optional): The number of steps to take before starting to train. Defaults to 1000.
            gradient_updates (int, optional): The number of gradient steps to take per update. Defaults to 1.
            project_name (str, optional): The name of the project. Defaults to "MORL Baselines".
            experiment_name (str, optional): The name of the experiment. Defaults to "GPI-PD Continuous Action".
            wandb_entity (Optional[str], optional): The wandb entity. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): The seed to use. Defaults to None.
            device (Union[th.device, str], optional): The device to use for training. Defaults to "auto".
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)
        # 保存所有超参数
        self.learning_rate = learning_rate
        self.tau = tau  # 软更新系数（Polyak平均）
        self.gamma = gamma  # 折扣因子
        self.buffer_size = buffer_size
        self.num_q_nets = num_q_nets  # 并行Q网络数量
        self.net_arch = net_arch
        self.learning_starts = learning_starts  # 开始训练的步数阈值
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates  # 每次更新的梯度步数
        self.alpha = alpha  # 熵系数

        self.replay_buffer = ReplayMemory(self.buffer_size)

        # 初始化多个Q网络（SAC使用多个Q网络来减少值高估）
        self.q_nets = [
            QNetwork(self.observation_dim, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            for _ in range(num_q_nets)
        ]
        # 初始化目标Q网络（用于稳定性）
        self.target_q_nets = [
            QNetwork(self.observation_dim, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            for _ in range(num_q_nets)
        ]
        # 目标网络初始化为Q网络的副本，但不参与梯度更新
        for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
            target_q_net.load_state_dict(q_net.state_dict())
            for param in target_q_net.parameters():
                param.requires_grad = False

        # 初始化权向量条件化策略
        self.policy = Policy(
            self.observation_dim, self.reward_dim, self.action_dim, self.env.action_space, net_arch=net_arch
        ).to(self.device)

        # Q网络优化器
        # *将列表解包为多个独立的迭代器参数，
        # itertools.chain的作用是将多个独立的参数迭代器拼接为一个连续的迭代器
        self.q_optim = optim.Adam(chain(*[net.parameters() for net in self.q_nets]), lr=self.learning_rate)
        # 策略优化器
        # list(...)：将参数迭代器转换为列表（非必需，Adam 优化器可直接接收迭代器，此处为代码可读性 / 兼容性写法）；
        self.policy_optim = optim.Adam(list(self.policy.parameters()), lr=self.learning_rate)

        self._n_updates = 0

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self):
        """Get the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "num_q_nets": self.num_q_nets,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "alpha": self.alpha,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def save(self, save_dir="weights/", filename=None, save_replay_buffer=True):
        """Save the agent's weights and replay buffer."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
        }
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            saved_params["q_net_" + str(i) + "_state_dict"] = q_net.state_dict()
            saved_params["target_q_net_" + str(i) + "_state_dict"] = target_q_net.state_dict()
        saved_params["q_nets_optimizer_state_dict"] = self.q_optim.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        """Load the agent weights from a file."""
        params = th.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(params["policy_state_dict"])
        self.policy_optim.load_state_dict(params["policy_optimizer_state_dict"])
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            q_net.load_state_dict(params["q_net_" + str(i) + "_state_dict"])
            target_q_net.load_state_dict(params["target_q_net_" + str(i) + "_state_dict"])
        self.q_optim.load_state_dict(params["q_nets_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def update(self):
        """Update the policy and the Q-nets."""
        for _ in range(self.gradient_updates):
            # 从缓存采样批次数据
            (s_obs, s_actions, w, s_rewards, s_next_obs, s_dones) = self._sample_batch_experiences()

            # ========== Q网络（批评家）更新 ==========
            with th.no_grad():
                # 用目标策略在下一状态采样动作
                next_actions, log_pi, _ = self.policy.sample(s_next_obs, w)
                # 用所有目标Q网络评估下一状态-动作对
                # 单个q_target(...)输出：因是批量处理，标量输出会自动扩展为形状(B, 1)的二维张量
                # th.stack(..., dim=0)：在第 0 维（网络维度） 拼接 num_q_nets 个(B,1)的张量，最终q_targets形状为 (num_q_nets, B, 1)
                q_targets = th.stack([q_target(s_next_obs, next_actions, w) for q_target in self.target_q_nets])
                # 取最小Q值（SAC特性：减少Q值高估）并减去熵项
                # (num_q_nets, B, 1) → 沿 dim=0 取最小后，消去第 0 维，得到形状 (B, 1) 的张量；
                # th.min(...)返回元组(最小值张量, 最小值索引)，[0]表示只取最小值张量，丢弃索引
                min_target_q = th.min(q_targets, dim=0)[0] - self.alpha * log_pi.reshape(-1, 1) # 原log_pi为(B,)（一维）→ 重塑后为 (B, 1)（二维）；

                # 计算Bellman目标：r + γ(1-done) * min_Q(s', a')
                target_q = (s_rewards + (1 - s_dones.reshape(-1, 1)) * self.gamma * min_target_q).detach()

            # 计算Q值预测
            q_values = [q_net(s_obs, s_actions, w) for q_net in self.q_nets]
            # Q网络损失：平均MSE
            critic_loss = (1 / self.num_q_nets) * sum([F.mse_loss(q_value, target_q) for q_value in q_values])

            # Q网络反向传播
            self.q_optim.zero_grad()
            critic_loss.backward()
            self.q_optim.step()

            # ========== 策略（演员）更新 ==========
            # 从当前策略采样动作
            pi, log_pi, _ = self.policy.sample(s_obs, w)
            # 评估采样动作的Q值
            q_values = th.stack([q_target(s_obs, pi, w) for q_target in self.q_nets])
            min_q = th.min(q_values, dim=0)[0]

            # 在多目标设置中，Q值乘以权向量求和得到加权回报
            # 这允许通过不同权向量获得帕累托前沿上的不同策略
            min_q = (min_q * w).sum(dim=-1, keepdim=True)
            # 策略损失：最小化熵项并最大化Q值
            policy_loss = ((self.alpha * log_pi) - min_q).mean()

            # 策略反向传播
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # ========== 目标网络软更新 ==========
            # 使用Polyak平均：target = (1-tau)*target + tau*current
            for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
                polyak_update(q_net.parameters(), target_q_net.parameters(), self.tau)

        if self.log and self.global_step % 100 == 0:
            wandb.log(
                {
                    "losses/critic_loss": critic_loss.item(),
                    "losses/policy_loss": policy_loss.item(),
                    "global_step": self.global_step,
                },
            )

    @th.no_grad()
    def eval(
        self, obs: Union[np.ndarray, th.Tensor], w: Union[np.ndarray, th.Tensor], torch_action=False
    ) -> Union[np.ndarray, th.Tensor]:
        """Evaluate the policy action for the given observation and weight vector."""
        # 将numpy输入转换为PyTorch张量
        if isinstance(obs, np.ndarray):
            obs = th.tensor(obs).float().to(self.device)
            w = th.tensor(w).float().to(self.device)

        # 获取确定性动作（评估模式）
        action = self.policy.get_action(obs, w)

        # 根据需要转换回numpy数组
        if not torch_action:
            action = action.detach().cpu().numpy()

        return action

    def train(
        self,
        total_timesteps: int,
        eval_env: gymnasium.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        eval_freq: int = 10000,
        reset_num_timesteps: bool = False,
        checkpoints: bool = False,
        save_freq: int = 10000,
    ):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps to train the agent for.
            eval_env (gym.Env): Environment to use for evaluation.
            ref_point (np.ndarray): Reference point for hypervolume calculation.
            known_pareto_front (Optional[List[np.ndarray]]): Optimal Pareto front, if known.
            num_eval_weights_for_front (int): Number of weights to evaluate for the Pareto front.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            eval_freq (int): Number of timesteps between evaluations during an iteration.
            reset_num_timesteps (bool): Whether to reset the number of timesteps.
            checkpoints (bool): Whether to save checkpoints.
            save_freq (int): Number of timesteps between checkpoints.
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "eval_freq": eval_freq,
                    "reset_num_timesteps": reset_num_timesteps,
                }
            )

        # 生成用于评估的权向量集合（均匀分布在权向量单纯形上）
        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        # 初始化权向量采样器（在参考方向周围以22.5°角范围采样）
        angle = th.pi * (22.5 / 180)
        weight_sampler = WeightSamplerAngle(self.env.unwrapped.reward_dim, angle)

        # 初始化训练计数器
        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        # 主训练循环
        obs, info = self.env.reset()
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            # 采样一个权向量
            tensor_w = weight_sampler.sample(1).view(-1).to(self.device)
            w = tensor_w.detach().cpu().numpy()

            # 学习初期采样随机动作以增加探索性
            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                # 学习阶段使用学习的策略
                with th.no_grad():
                    action = self.policy.get_action(
                        th.tensor(obs).float().to(self.device),
                        tensor_w,
                    )
                    action = action.detach().cpu().numpy()

            action_env = action

            # 执行环境步进
            next_obs, vector_reward, terminated, truncated, info = self.env.step(action_env)

            # 将转移存入回放缓存
            self.replay_buffer.push(obs, action, w, vector_reward, next_obs, terminated)

            # 梯度更新
            if self.global_step >= self.learning_starts:
                self.update()

            # 处理集结束
            if terminated or truncated:
                obs, _ = self.env.reset()
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, w, self.global_step)
            else:
                obs = next_obs

            # 周期性评估
            if self.log and self.global_step % eval_freq == 0:
                # 在所有评估权向量下测试策略
                returns_test_tasks = [
                    policy_evaluation_mo(self, eval_env, ew, rep=num_eval_episodes_for_front)[3] for ew in eval_weights
                ]
                # 记录多目标评估指标（超体积、期望效用等）
                log_all_multi_policy_metrics(
                    current_front=returns_test_tasks,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                )

            # 定期保存检查点
            if checkpoints and self.global_step % save_freq == 0:
                self.save(filename=f"CAPQL step={self.global_step}", save_replay_buffer=False)

        self.close_wandb()
