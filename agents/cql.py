from agents import sac
import time
import torch,gym
import numpy as np
from rlkit_cql.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit_cql.torch.networks import FlattenMlp

import rlkit_cql.torch.pytorch_util as ptu
from rlkit_cql.data_management.env_replay_buffer import EnvReplayBuffer
import d4rl
from rlkit_cql.torch.core import np_to_pytorch_batch
from torch import nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hdf5(dataset, replay_buffer):
    replay_buffer._size = dataset['terminals'].shape[0]

    replay_buffer._observations[:replay_buffer._size] = dataset['observations']
    replay_buffer._next_obs[:replay_buffer._size] = dataset['next_observations']
    replay_buffer._actions[:replay_buffer._size] = dataset['actions']
    replay_buffer._rewards[:replay_buffer._size] = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals[:replay_buffer._size] = np.expand_dims(np.squeeze(dataset['terminals']), 1)  
    # replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


class ReplayBuffer(sac.ReplayBuffer):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        super().__init__(obs_dim, act_dim, size)

    def sample(self, batch_size=100):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


class agent(sac.agent):
    def __init__(self, args, logger, replay_size=int(2e6), policy_lr=1E-4, qf_lr=3E-4, lagrange_thresh=-1.0, optimizer_class=torch.optim.Adam):
        super().__init__(args, logger)
        self.num_random = 10
        self.discrete = False
        self.temp = 1.0
        self.reward_scale = 1
        self.min_q_weight = 5.0
        self.policy_eval_start = 40000
        # ptu.set_gpu_mode(True)
        self.qf_criterion = nn.MSELoss()

        eval_env = gym.make(args.env)
        expl_env = eval_env
        M = 256

        self.explore_env = gym.make(args.env)
        self.explore_env.seed(self.args.seed)
        
        obs_dim = expl_env.observation_space.low.size
        action_dim = eval_env.action_space.low.size

        self.qf1 = FlattenMlp( 
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            ).to(device)
        self.qf2 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            ).to(device)
        self.target_qf1 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            ).to(device)
        self.target_qf2 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            ).to(device)

        self.policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M], 
        ).to(device)
        
        self.eval_policy = MakeDeterministic(self.policy).to(device)

        self.replay_buffer = EnvReplayBuffer(
        int(3E6),
        expl_env,
        )
        load_hdf5(d4rl.qlearning_dataset(eval_env), self.replay_buffer)
  
        self.with_lagrange = False # if lagrange_thresh < 0.0:
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )

        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        self.log_alpha = zeros(1, requires_grad=True)
        self.alpha_optimizer = optimizer_class(
            [self.log_alpha],
            lr=policy_lr,
        )

        self.pi_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.q1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.q2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)


    def populate_replay_buffer(self):
        dataset = d4rl.qlearning_dataset(self.env)
        self.replay_buffer.obs_buf[:dataset['observations'].shape[0],:] = dataset['observations']
        self.replay_buffer.act_buf[:dataset['actions'].shape[0],:] = dataset['actions']
        self.replay_buffer.obs2_buf[:dataset['next_observations'].shape[0],:] = dataset['next_observations']
        self.replay_buffer.rew_buf[:dataset['rewards'].shape[0]] = dataset['rewards']
        self.replay_buffer.done_buf[:dataset['terminals'].shape[0]] = dataset['terminals']
        self.replay_buffer.size = dataset['observations'].shape[0]
        self.replay_buffer.ptr = (self.replay_buffer.size+1)%(self.replay_buffer.max_size)

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp, reparameterize=True, return_log_prob=True,
        )
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def compute_loss_q(self, data, gamma=0.99):
        obs, actions, rewards, next_obs, terminals = data['observations'], data[
            'actions'], data['rewards'], data['next_observations'], data['terminals']
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        # with torch.no_grad():
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, return_log_prob=True, # reparameterize=True, 
        )
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
            obs,  return_log_prob=True, # reparameterize=True,
        )

        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions)
        )

        q_target = self.reward_scale * rewards + (1. - terminals) * gamma * target_q_values
        q_target = q_target.detach()

        # MSE loss against Bellman backup
        qf1_loss = self.qf_criterion(q1_pred,q_target)
        qf2_loss = self.qf_criterion(q2_pred,q_target)

        # add CQL
        random_actions_tensor = torch.FloatTensor(
            q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).cuda()
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(
            obs, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(
            next_obs, num_actions=self.num_random, network=self.policy)
        q1_rand = self._get_tensor_values(
            obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(
            obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, network=self.qf2)

        # importance sammpled version
        random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
        cat_q1 = torch.cat(
            [q1_rand - random_density, q1_next_actions -
                new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
        )
        cat_q2 = torch.cat(
            [q2_rand - random_density, q2_next_actions -
                new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
        )

        min_qf1_loss = torch.logsumexp(
            cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(
            cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        return qf1_loss, qf2_loss

    def explore(self,online_explore_steps,max_ep_len=1000):
        o, ep_ret, ep_len = self.explore_env.reset(), 0, 0
        for _ in range(online_explore_steps):
            a, *_ = self.policy.get_action(o)
            o2, r, d, _ = self.explore_env.step(a)               
            ep_ret += r
            ep_len += 1
            d = False if ep_len==max_ep_len else d
            self.replay_buffer.add_sample(o, a, r, d, o2,env_info={})
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                self.logger.store(EpRet=int(ep_ret), EpLen=ep_len)
                o, ep_ret, ep_len = self.explore_env.reset(), 0, 0 

    def update_q(self, data):
        q1_loss, q2_loss = self.compute_loss_q(data)

        self.q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.q2_optimizer.step()

    def compute_loss_pi(self, data):
        o = data['observations'] # 'obs'
        # # automatic entropy tuning
        pi, policy_mean, policy_log_std, log_pi, *_ = self.policy(
        o, reparameterize=True, return_log_prob=True)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        q1_pi = self.qf1(o, pi)
        q2_pi = self.qf2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * log_pi - q_pi).mean()
        
        pi_info = {}

        self.logger.store(logp_pi_t=log_pi.mean().item())

        return loss_pi, pi_info

    def test_policy(self, eval_episodes=5):
        # self.policy.cpu()
        eval_env = gym.make(self.args.env)
        eval_env.seed(self.args.seed + 100)
        avg_reward = 0.
        
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                # state = torch.FloatTensor(state).to(device)
                action,_ = self.eval_policy.get_action(state)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes
        return 100*eval_env.get_normalized_score(avg_reward),avg_reward


    def train(self, polyak=0.995, ue_loss_k=1000, ue_rollout=1000, ue_train_epoch=50, augment_mc='gain'):
        time_start = time.time()

        for t in range(int(self.args.max_timesteps)):
            np_data = self.replay_buffer.random_batch(self.args.batch_size)
            data = np_to_pytorch_batch(np_data)
  
            loss_pi, pi_info = self.compute_loss_pi(data)
            q1_loss, q2_loss = self.compute_loss_q(data)

            self.q1_optimizer.zero_grad()
            q1_loss.backward(retain_graph=True)
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward(retain_graph=True)
            self.q2_optimizer.step()

            self.pi_optimizer.zero_grad()
            loss_pi.backward(retain_graph=False)
            self.pi_optimizer.step()

            with torch.no_grad():
                for p, p_targ in zip(self.qf1.parameters(), self.target_qf1.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.qf2.parameters(), self.target_qf2.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

            if t % self.args.eval_freq == 0:  # eval_freq should be 1000
                norm_score, test_score = self.test_policy(eval_episodes=5)
                self.logger.store(TestScore=int(test_score))
                self.logger.store(NormScore=int(norm_score))

                if t % self.args.print_freq == 0:  # print every 3000 steps
                    self.logger.save_print(f"self.alpha:{self.alpha}")
                    if not ("debug" in self.args.version):
                        self.writer.add_scalar(
                            "TestScore", self.logger.get_stats("TestScore")[0], t)

                    self.logger.log_tabular("TimeSteps", t)
                    self.logger.log_tabular("TestScore", average_only=True)
                    self.logger.log_tabular("NormScore",average_only=True)
                    self.logger.log_tabular("Time", int(time.time()-time_start))
                    self.logger.log_tabular("logp_pi_t", average_only=True)
                    self.logger.dump_tabular()

        self.writer.flush()
        self.writer.close()



    def train_offline(self, online_explore_steps,offline_steps, polyak=0.995, ue_loss_k=1000, ue_rollout=1000, ue_train_epoch=50, augment_mc='gain'):
        time_start = time.time()

        for t in range(int(offline_steps)):
            np_data = self.replay_buffer.random_batch(self.args.batch_size)
            data = np_to_pytorch_batch(np_data)
  
            loss_pi, pi_info = self.compute_loss_pi(data)
            q1_loss, q2_loss = self.compute_loss_q(data)

            self.q1_optimizer.zero_grad()
            q1_loss.backward(retain_graph=True)
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward(retain_graph=True)
            self.q2_optimizer.step()

            self.pi_optimizer.zero_grad()
            loss_pi.backward(retain_graph=False)
            self.pi_optimizer.step()

            with torch.no_grad():
                for p, p_targ in zip(self.qf1.parameters(), self.target_qf1.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.qf2.parameters(), self.target_qf2.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

        self.writer.flush()
        self.writer.close()