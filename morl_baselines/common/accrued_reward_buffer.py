"""Accrued reward buffer for ESR algorithms."""

import numpy as np
import torch as th


class AccruedRewardReplayBuffer:
    """Replay buffer with accrued rewards stored (for ESR algorithms)."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        rew_dim=1,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
    ):
        """Initialize the Replay Buffer.

        Args:
            obs_shape: Shape of the observations
            action_shape:  Shape of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
        """
        self.max_size = max_size
        # 循环队列的队头指针和队列大小
        self.ptr, self.size = 0, 0
        # obs_shape 和 action_shape 是 (obs_dim,) 和 (action_dim,) 的形式
        # (max_size,) + obs_shape = (max_size, obs_dim)
        # (max_size,) + action_shape = (max_size, action_dim)
        self.obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((max_size,) + action_shape, dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.accrued_rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, accrued_reward, action, reward, next_obs, done):
        """Add a new experience to memory.

        Args:
            obs: Observation
            accrued_reward: Accrued reward
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done
        """
        # obs 有可能是标量或列表，其他变量取决于 gym 环境
        # 假设 obs 是一个 numpy 数组，则 np.array(obs) 可能会共享内存
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.accrued_rewards[self.ptr] = np.array(accrued_reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        """Sample a batch of experiences.

        Args:
            batch_size: Number of elements to sample
            replace: Whether to sample with replacement or not
            use_cer: Whether to use CER or not
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        # 从 [0, size) 中随机采样 batch_size 个索引，replace=replace 表示是否允许重复采样
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience
        experience_tuples = (
            self.obs[inds],
            self.accrued_rewards[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor: # 将元组的每个元素转换为张量
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def cleanup(self):
        """Cleanup the buffer."""
        self.size, self.ptr = 0, 0

    def get_all_data(self, to_tensor=False, device=None):
        """Returns the whole buffer.

        Args:
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        inds = np.arange(self.size)
        experience_tuples = (
            self.obs[inds],
            self.accrued_rewards[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size
