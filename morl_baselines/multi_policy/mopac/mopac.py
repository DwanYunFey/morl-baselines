
import time
import os
from copy import deepcopy
from typing import Optional, List, Tuple, Union
from typing_extensions import override

from distutils.util import strtobool

import math
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

from mopac.architectures import MOCriticNetEpistemic, MOActorNetProbabilistic

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import log_episode_info, log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.pareto import ParetoArchive


class MOPAC(MOPolicy):
    """Multi-objective Probabilistic Actor-Critic (PAC) algorithm.

    It is a multi-objective version of the PAC algorithm, with multi-objective critic and weighted sum scalarization.
    """

    def __init__(
        self,
        env: gym.Env,
        learn_frequency: int,
        eval_episodes: int,
        warmup_steps: int,
        ref_point: np.ndarray,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        n_hidden: int = 256,
        learning_rate: float = 3e-4,
        q_lr: float = 1e-3,
        policy_freq: int = 2,
        target_net_freq: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
        id: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        project_name: str = "MORL-Baselines",
        experiment_name: str = "MO-PAC",
        wandb_entity: Optional[str] = None,
        ref_front: Optional[List[np.ndarray]] = None,
        log: bool = True,
        seed: int = 42,
        parent_rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the MOPAC algorithm.

        Args:
            env: Env
            weights: weights for the scalarization
            buffer_size: buffer size
            gamma: discount factor
            tau: target smoothing coefficient (polyak update)
            batch_size: batch size
            warmup_steps: how many steps to collect before triggering the learning
            n_hidden: number of nodes in the hidden layers
            learning_rate: learning rate of the policy
            q_lr: learning rate of the q networks
            policy_freq: the frequency of training policy (delayed)
            target_net_freq: the frequency of updates for the target networks
            alpha: Entropy regularization coefficient
            autotune: automatic tuning of alpha
            id: id of the SAC policy, for multi-policy algos
            device: torch device
            torch_deterministic: whether to use deterministic version of pytorch
            log: logging activated or not
            seed: seed for the random generators
            parent_rng: parent random generator, for multi-policy algos
        """
        super().__init__(id, device)
        # Seeding
        self.seed = seed
        self.parent_rng = parent_rng
        if parent_rng is not None:
            self.np_random = parent_rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # env setup
        self.env = env
        assert isinstance(self.env.action_space, gym.spaces.Box), "only continuous action space is supported"
        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.reward_dim = self.env.unwrapped.reward_space.shape[0]

        # Scalarization
        self.weights = np.ones(self.reward_dim) / self.reward_dim
        self.weights_tensor = th.from_numpy(self.weights).float().to(self.device)
        self.batch_size = batch_size

        self.recent_rewards = []  
        self.N_recent = 50

        # PAC Parameters
        self.learn_frequency = learn_frequency
        self.eval_episodes = eval_episodes

        self.buffer_size = buffer_size
        self.alpha = alpha
        self.delta = 0.05
        self.gamma = gamma
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.q_lr = q_lr
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq
        
        self.actor_var_upper_clamp=-7.0
        self._critic_var_upper_clamp = 4.0

        # Networks
        
        self.critic = []
        self.critic_t = []
        self.critic_optim = []
        self.critic_scheduler = []
        
        for i in range(self.policy_freq):
            self.critic.append(MOCriticNetEpistemic(self.obs_shape, self.action_shape, self.reward_dim, self.n_hidden).to(self.device))
            self.critic_t.append(MOCriticNetEpistemic(self.obs_shape, self.action_shape, self.reward_dim, self.n_hidden).to(self.device))
            optim_i = th.optim.Adam(self.critic[i].parameters(), self.learning_rate)
            self.critic_optim.append(optim_i)
            self.critic_scheduler.append(th.optim.lr_scheduler.LinearLR(optimizer=optim_i, start_factor=1, end_factor=0.1,total_iters=1000))
            self._hard_update(self.critic[i], self.critic_t[i]) # hard update at the beginning

        self._actor = MOActorNetProbabilistic(self.obs_shape, self.action_shape, self.actor_var_upper_clamp).to(self.device)
        self._actor_optim = th.optim.Adam(self._actor.parameters(), self.learning_rate)
        self._actor_scheduler = th.optim.lr_scheduler.LinearLR(optimizer=self._actor_optim, start_factor=1, end_factor=0.1,total_iters=1000)


        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -th.prod(th.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            # self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = alpha
        self.alpha_tensor = th.scalar_tensor(self.alpha).to(self.device)

        # Buffer
        self.env.observation_space.dtype = np.float32
        self.buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=self.action_shape[0],
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
        )

        # Logging
        self.global_step = 0
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.ref_point = ref_point
        self.ref_front = ref_front
        self.n_sample_weights = 50
        self.archive = ParetoArchive()

        
        
        if self.log:
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name, entity=wandb_entity)

    def setup_wandb(
        self, project_name: str, experiment_name: str, entity: Optional[str] = None, group: Optional[str] = None
    ) -> None:
        """Initializes the wandb writer.

        Args:
            project_name: name of the wandb project. Usually MORL-Baselines.
            experiment_name: name of the wandb experiment. Usually the algorithm name.
            entity: wandb entity. Usually your username but useful for reporting other places such as openrlbenmark.

        Returns:
            None
        """
        self.experiment_name = experiment_name
        env_id = self.env.spec.id if not isinstance(self.env, MOSyncVectorEnv) else self.env.envs[0].spec.id
        self.full_experiment_name = f"{env_id}__{experiment_name}__{self.seed}__{int(time.time())}"
        import wandb

        config = self.get_config()
        config["algo"] = self.experiment_name
        # looks for whether we're using a Gymnasium based env in env_variable
        monitor_gym = strtobool(os.environ.get("MONITOR_GYM", "True"))

        wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            name=self.full_experiment_name,
            monitor_gym=monitor_gym,
            save_code=True,
            group=group,
        )
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self) -> None:
        """Closes the wandb writer and finishes the run."""
        import wandb

        wandb.finish()

    def get_config(self) -> dict:
        """Returns the configuration of the policy."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "n_hidden": self.n_hidden,
            "learning_rate": self.learning_rate,
            "q_lr": self.q_lr,
            "policy_freq": self.policy_freq,
            "target_net_freq": self.target_net_freq,
            "alpha": self.alpha,
            "autotune": self.autotune,
            "seed": self.seed,
        }

    def __deepcopy__(self, memo):
        """Deep copy of the policy.

        Args:
            memo (dict): memoization dict
        """
        copied = type(self)(
            env=self.env,
            weights=self.weights,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            tau=self.tau,
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            n_hidden=self.n_hidden,
            learning_rate=self.learning_rate,
            q_lr=self.q_lr,
            policy_freq=self.policy_freq,
            target_net_freq=self.target_net_freq,
            alpha=self.alpha,
            autotune=self.autotune,
            id=self.id,
            device=self.device,
            log=self.log,
            seed=self.seed,
            parent_rng=self.parent_rng,
        )

        # Copying networks
        copied._actor = deepcopy(self._actor)
        copied.critic = deepcopy(self.critic)
        copied.critic_t = deepcopy(self.critic_t)

        copied.global_step = self.global_step
        copied._actor_optim = optim.Adam(copied._actor.parameters(), lr=self.learning_rate, eps=1e-5)
        copied.critic_optim = optim.Adam(list(copied.critic_optim[0].parameters()) + list(copied.critic_optim[1].parameters()), lr=self.q_lr)
        if self.autotune:
            copied.a_optimizer = optim.Adam([copied.log_alpha], lr=self.q_lr)
        copied.alpha_tensor = th.scalar_tensor(copied.alpha).to(self.device)
        copied.buffer = deepcopy(self.buffer)
        return copied

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def _hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    @override
    def get_buffer(self):
        return self.buffer

    @override
    def set_buffer(self, buffer):
        self.buffer = buffer

    @override
    def get_policy_net(self) -> th.nn.Module:
        return self._actor

    @override
    def set_weights(self, weights: np.ndarray):
        self.weights = weights
        self.weights_tensor = th.from_numpy(self.weights).float().to(self.device)

    
    def KL(self, mu1, mu2, logvar1, logvar2):
        logvar1 = logvar1.clamp(-10,self._critic_var_upper_clamp)
        logvar2 = logvar2.clamp(-10,self._critic_var_upper_clamp)
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        return 0.5*(logvar2 - logvar1) + (var1 + (mu1-mu2)**2) / (2.0 * var2) - 0.5

    def compute_Q_target(self, s, a, r, sp, done):
        with th.no_grad():
            ap_pred, _ = self._actor(sp, is_training=True)
            qp_t_mu_list, qp_t_logvar_list = [], []
            
            for j in range(self.policy_freq):
                qp_i_t = self.critic_t[j](sp, ap_pred)
                qp_i_t_mu = qp_i_t[:, :self.reward_dim]  
                qp_i_t_logvar = qp_i_t[:, self.reward_dim:2*self.reward_dim]  
                qp_t_mu_list.append(qp_i_t_mu)
                qp_t_logvar_list.append(qp_i_t_logvar)

            qp_t_mu_list = th.stack(qp_t_mu_list, dim=-1)  # (batch_size, num_objectives, ensemble_size)
            qp_t_logvar_list = th.stack(qp_t_logvar_list, dim=-1)  # (batch_size, num_objectives, ensemble_size)

            idx = qp_t_mu_list.argmin(dim=-1, keepdim=True).expand(-1, self.reward_dim, -1)
            qp_t_logvar = th.gather(qp_t_logvar_list, -1, idx).squeeze(-1)
            qp_t_mu = th.gather(qp_t_mu_list, -1, idx).squeeze(-1)

            q_t_logvar = 2.0 * math.log(self.gamma) + qp_t_logvar
            q_t_mu = r + (self.gamma * qp_t_mu * (1 - done.unsqueeze(-1)))

            return q_t_mu, q_t_logvar


    def Q_eval(self, s, a, critic_list):
        q_pi_mu_list, q_pi_logvar_list = [], []

        for i in range(self.policy_freq):
            q_pi_params = critic_list[i](s, a)
            q_pi_mu = q_pi_params[:, :self.reward_dim]  # (batch_size, num_objectives)
            q_pi_logvar = q_pi_params[:, self.reward_dim:2*self.reward_dim]  # (batch_size, num_objectives)
            q_pi_mu_list.append(q_pi_mu)
            q_pi_logvar_list.append(q_pi_logvar)

        q_pi_mu_list = th.stack(q_pi_mu_list, dim=-1)  # (batch_size, num_objectives, ensemble_size)
        q_pi_logvar_list = th.stack(q_pi_logvar_list, dim=-1)  # (batch_size, num_objectives, ensemble_size)

        idx = q_pi_mu_list.argmin(dim=-1, keepdim=True).expand(-1, self.reward_dim, -1)
        q_pi_logvar = th.gather(q_pi_logvar_list, -1, idx).squeeze(-1)
        q_pi_mu = th.gather(q_pi_mu_list, -1, idx).squeeze(-1)

        return q_pi_mu, q_pi_logvar.clamp(-10, self._critic_var_upper_clamp).exp()

    
    @th.no_grad()
    def select_action(self, s, is_training=True, method="nopareto"):
        s = th.from_numpy(s).view(1,-1).float().to(self.device)
        a_dist, _ = self._actor(s, is_training=is_training)
        a = a_dist.cpu().numpy().squeeze(0)
        return a


    @th.no_grad()
    def Q_values(self, s, a):
        
        s = th.from_numpy(s).view(1, -1).float().to(self.device)
        a = th.from_numpy(a).view(1,-1).float().to(self.device)

        q = self.critic[0](s, a)  # The critic here needs to output the Q values of multiple targets
        
        q_mu = q[ :, :self.reward_dim]  # Take the Q values of all targets
        return q_mu.cpu().numpy()
    
    @th.no_grad()
    def compute_dynamic_weights(self):
        """
        Calculate dynamic weights using the avd_reward (array of shape: (reward-dim)) from the last N episodes.
        Return: a numpy array with the shape (reward_im,), The sum of weights is 1.
        """
        if len(self.recent_rewards) == 0:
            #Return uniform weight when there is no data
            return np.ones(self.reward_dim) / self.reward_dim

        #Collect recent rewards into a matrix with the shape (N_recent, reward_dem)
        recent = np.stack(self.recent_rewards, axis=0)
        #Calculate the average reward for each target


        # avg_per_objective = recent.mean(axis=0)  # shape: (reward_dim,)
        # #Assuming that lower target rewards require higher weights, the following method can be used:
        # # 1.  Find the reciprocal of each objective (add a small constant to prevent division by zero)
        # inv_rewards = 1.0 / (avg_per_objective + 1e-8)
        # # 2.  normalization
        # dynamic_weights = inv_rewards / np.sum(inv_rewards)

        smooth_rewards = np.mean(recent, axis=0)
        dynamic_weights = np.exp(smooth_rewards) / np.sum(np.exp(smooth_rewards))
        
        return dynamic_weights

    @th.no_grad()
    def select_action_dynamic(self, s, is_training=False, method="dynamic"):
        #Convert the state to tensor, shape (1, obs-dim)
        s = th.from_numpy(s).view(1, -1).float().to(self.device)
        
        #Obtain action distribution
        a_dist = self._actor(s, is_training=is_training, return_distribution = True)
        
        if method == "dynamic":
            num_samples = 50
            #Sample multiple candidate actions from the action distribution:
            #The shape of a_samples is (num_stamples, 1, action-dim)
            a_samples = a_dist.rsample((num_samples,))
            #Remove the batch dimension and become (num_stamples, action-dim)
            a_samples = a_samples.squeeze(1)
            
            #Use critic to evaluate these candidate actions
            #Q-values_multi can accept s and a_samples, return the shape (num_stamples, reward_im)
            q_values = self.Q_values_multi(s.cpu().numpy(), a_samples.cpu().numpy())
            
            #Calculate the dynamic weight w (for example, call compute_dynamics_ceights() to obtain shape (reward_im,))
            w = self.compute_dynamic_weights()  #User defined function, calculated based on the most recent evaluation reward
            
            #Calculate the comprehensive utility for each candidate action by simply weighting and summing the Q-values of each target
            #The shape of candidate_util is (num_stamples,)
            candidate_util = q_values.dot(w)
            
            #Select the candidate action with the highest comprehensive utility
            best_idx = np.argmax(candidate_util)
            a = a_samples[best_idx]
            a = a.cpu().numpy()
        else:
            a = a_dist.cpu().numpy().squeeze(0)
        
        return a

    @th.no_grad()
    def Q_values_multi(self, s, a_samples):
        """
        Calculate the Q-values of multiple candidate actions in a state (multi-objective).
                
        Args:
        s: Numpy array, shape (1, obs_im) - single state
        A_samples: numpy array, shape (num_stamples, action-dim) - candidate action set
                    
        Returns:
        Q_mu: numpy array, shape (num_stamples, reward_im), representing the multi-objective Q-values (mean part) corresponding to each candidate action
        """
        #Convert s to Tensor, Shape (1, obs_im)
        s_tensor = th.from_numpy(s).float().to(self.device)
        #Convert a_samples to tensors, shapes (num_stamples, action-dim)
        a_tensor = th.from_numpy(a_samples).float().to(self.device)
        
        #Repeat s_tensor num_stamples multiple times to obtain the shape (num_stamples, obs-dim)
        s_repeated = s_tensor.repeat(a_tensor.shape[0], 1)
        
        #Call the critic network, input the state and candidate actions
        #The output shape of the critic network should be (num_stamples, 2 * reward_im)
        q = self.critic[0](s_repeated, a_tensor)
        #Extract the Q-value mean part: front reward_im column
        q_mu = q[:, :self.reward_dim]
        
        return q_mu.cpu().detach().numpy()

    
    @override
    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None) -> Union[int, np.ndarray]:
        """Returns the best action to perform for the given obs.

        Args:
            obs: observation as a numpy array
            w: None
        Return:
            action as a numpy array (continuous actions)
        """
        obs = th.as_tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        with th.no_grad():
            action, _, _ = self.actor.get_action(obs)

        return action[0].detach().cpu().numpy()

    @override
    def update(self):
        
        (mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = self.buffer.sample(
            self.batch_size, to_tensor=True, device=self.device
        )

        q_t_mu, q_t_logvar = self.compute_Q_target(mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones)

        # update critic ensemble
        for i in range(self.policy_freq):
            self.critic_optim[i].zero_grad()
            q = self.critic[i](mb_obs, mb_act)
            q_mu = q[:, :self.reward_dim]
            q_logvar = q[:, self.reward_dim:2*self.reward_dim]
            q_var = q_logvar.clamp(-10,self._critic_var_upper_clamp).exp()

            # ------------  1 -----------------
            # Calculate PAC Bayes loss
            t1 = 1.0/th.sqrt(1.0+2.0*q_var)
            pre_loss = (q_mu-q_t_mu)**2
            divisor = (2.0*q_var+1)
            t2 = th.exp(-pre_loss/divisor)
            q_loss = (1.0 - t1*t2).mean().sum()

            # Calculate PAC Bayes confidence term
            num_observations = self.batch_size
            mu_prior = q_t_mu.clone().detach()
            logvar_prior = th.ones_like(mu_prior).to(self.device)*(-8)

            N = (th.ones(1)*num_observations).to(self.device).squeeze()
            confidence_term = th.log(2.0 * N.sqrt() / (self.delta))
            denominator = 2.0*N

            sqrt_term = ((self.KL(q_mu, mu_prior, q_logvar, logvar_prior).sum(dim=0) + confidence_term) / denominator).sqrt()
            q_loss += sqrt_term.sum()


            # ------------  2 -----------------
            # Calculate PAC Bayes loss
            t1 = 1.0 / th.sqrt(1.0 + 2.0 * q_var)  # (batch_size, reward_dim)
            pre_loss = (q_mu - q_t_mu) ** 2         # (batch_size, reward_dim)
            divisor = (2.0 * q_var + 1)             # (batch_size, reward_dim)
            t2 = th.exp(-pre_loss / divisor)        # (batch_size, reward_dim)
            # Calculate the mean of each target on a batch to obtain a vector (reward-dim,)
            per_target_loss = (1.0 - t1 * t2).mean(dim=0)

            # -----------------------------
            # Calculate PAC Bayes confidence term
            # -----------------------------
            num_observations = self.batch_size
            mu_prior = q_t_mu.clone().detach()
            logvar_prior = th.ones_like(mu_prior).to(self.device) * (-8)

            N = (th.ones(1) * num_observations).to(self.device).squeeze()
            confidence_term = th.log(2.0 * N.sqrt() / (self.delta))
            denominator = 2.0 * N

            # KL divergence is summed on each target, with a shape of (reward-dim,)
            KL_term = self.KL(q_mu, mu_prior, q_logvar, logvar_prior).sum(dim=0)
            sqrt_term = ((KL_term + confidence_term) / denominator).sqrt()  # (reward_dim,)

            # The ultimate loss for each target
            per_target_loss = per_target_loss + sqrt_term  # (reward_dim,)

            # -----------------------------
            # 融合动态权重调整
            # -----------------------------
            # 动态权重为 numpy 数组，转换为 tensor (reward_dim,)
            dynamic_weights = th.from_numpy(self.compute_dynamic_weights()).float().to(self.device)
            # 加权求和获得最终的critic损失（标量）
            q_loss = (per_target_loss * dynamic_weights).sum()
            # -----------------------------


            q_loss.backward()
            self.critic_optim[i].step()

            if self.global_step % 100 == 0 and self.log:
                log_str = f"_{self.id}" if self.id is not None else ""
                to_log = {
                    f"losses{log_str}/q_loss": q_loss.item(),
                    
                    f"metrics{log_str}/q_mu_mean": q_mu.mean().item(),
                    f"metrics{log_str}/q_logvar_mean": q_logvar.mean().item(),
                    f"metrics{log_str}/target_q_mu_mean": q_t_mu.mean().item(),
                    f"metrics{log_str}/target_q_logvar_mean": q_t_logvar.mean().item(),
                    "global_step": self.global_step,
                }
                
                if hasattr(self, "alpha"):
                    to_log[f"losses{log_str}/alpha"] = self.alpha
                wandb.log(to_log)

        
        # Linear Scalarization
        # weights_x = th.tensor([1.0 / self.reward_dim] * self.reward_dim).to(self.device)
        # weighted_q = (q_pi_mu + 0.5 * q_pi_var / self.alpha) * weights_x  # (batch_size, num_objectives)
        # pi_loss = (self.alpha * e_pred - weighted_q.sum(dim=-1)).mean()

        # update actor
        self._actor_optim.zero_grad()
        a_pred, e_pred = self._actor(mb_obs, is_training=True)
        q_pi_mu, q_pi_var = self.Q_eval(mb_obs, a_pred, self.critic)

        pi_loss = (self.alpha*e_pred - (q_pi_mu + 0.5*q_pi_var/self.alpha)).mean()

        pi_loss.backward()
        self._actor_optim.step()

        for i in range(self.policy_freq):
            self._soft_update(self.critic[i], self.critic_t[i])


    def eval_policy(self,
            eval_env: Optional[gym.Env] = None,
            w: Optional[np.ndarray] = None):
        """
        Args:
            obs: observation as a numpy array
            w: None
        """

        with th.no_grad():
            
            results = np.zeros((self.eval_episodes, self.reward_dim))
            avg_reward = np.zeros((self.eval_episodes, self.reward_dim))
            ref_point = self.ref_point
            reward_dim = self.reward_dim
            global_step= self.global_step
            n_sample_weights = self.n_sample_weights
            ref_front = self.ref_front

            for episode in range(self.eval_episodes):
                obs, _ = eval_env.reset()

                step = 0
                
                a = self.select_action_dynamic(obs, is_training=False)

                done = False

                while not done:
                    a = self.select_action_dynamic(obs, is_training=False)
                    sp, r, term, trunc, info = eval_env.step(a)
                    done = term or trunc
                    obs = sp
                    results[episode] += r
                    avg_reward[episode] += self.gamma**step * r
                    step += 1
                    
                self.recent_rewards.append(avg_reward[episode])
                if len(self.recent_rewards) > self.N_recent:
                    self.recent_rewards.pop(0)
                
                smooth_rewards = np.mean(self.recent_rewards, axis=0)
                self.dynamic_weights = np.exp(smooth_rewards) / np.sum(np.exp(smooth_rewards))
                

            if self.log:
                current_front = [np.array(r) for r in avg_reward]   
                log_all_multi_policy_metrics(
                    current_front,
                    ref_point,  
                    reward_dim,
                    global_step,
                    n_sample_weights,
                    ref_front  
                )


    def train(self, eval_env: Optional[gym.Env], total_timesteps: int, eval_frequency: int,  start_time=None):
        if start_time is None:
            start_time = time.time()
        
        obs, _ = self.env.reset()
        r_cum = np.zeros(self.reward_dim) 
        episode = 0
        e_step = 0

        while self.global_step < total_timesteps:
            e_step += 1

            if self.global_step % eval_frequency == 0:
                self.eval_policy(eval_env)

            # warmup 
            if self.global_step < self.warmup_steps:
                a = self.env.action_space.sample()
            else:
                a = self.select_action_dynamic(obs)
            a = np.clip(a, -1.0, 1.0)

            #  r : (m,)
            sp, r, done, truncated, info = self.env.step(a)
            self.buffer.add(obs=obs, next_obs=sp, action=a, reward=r, done=done)
            # self.agent.store_transition(s, a, r, sp, done, step + 1)

            obs = sp
            r_cum += r  

            # trainning
            if self.global_step >= self.warmup_steps and (self.global_step % self.learn_frequency) == 0:
                self.update()
                if self.log and self.global_step % 100 == 0:
                    print("SPS:", int(self.global_step / (time.time() - start_time)))
                    wandb.log(
                        {"charts/SPS": int(self.global_step / (time.time() - start_time)), "global_step": self.global_step}
                    )

            if done or truncated:
                print('Episode:', episode, ' Reward:', r_cum, ' global_step:', self.global_step)
                obs, _ = self.env.reset()
                r_cum = np.zeros(self.reward_dim)
                episode += 1
                e_step = 0
            
            self.global_step +=1

        self.eval_policy(eval_env)
        end_time = time.time()

        print("done!")
        self.env.close()
        eval_env.close()
        self.close_wandb()



