from agents import cql
import time
import torch,gym
import numpy as np
from rlkit_cql.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit_cql.torch.networks import FlattenMlp

import rlkit_cql.torch.pytorch_util as ptu
from rlkit_cql.data_management.env_replay_buffer import EnvReplayBuffer
import d4rl
from rlkit_cql.torch.core import np_to_pytorch_batch
from torch import dtype, nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class agent(cql.agent):
    def __init__(self, args, logger, replay_size=int(2e6), policy_lr=1E-4, qf_lr=3E-4, lagrange_thresh=-1.0, optimizer_class=torch.optim.Adam):
        super().__init__(args, logger)
        eval_env = gym.make(args.env)
        expl_env = eval_env
        M = 256
        ptu.set_gpu_mode(True)
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
        self.qf3 = FlattenMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                ).to(device)
        self.qf4 = FlattenMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                ).to(device)
        self.qf5 = FlattenMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                ).to(device)

        self.qf_list = [self.qf1,self.qf2,self.qf3,self.qf4,self.qf5]


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
        self.target_qf3 = FlattenMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                ).to(device)
        self.target_qf4 = FlattenMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                ).to(device)
        self.target_qf5 = FlattenMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                ).to(device) 

        self.target_qf_list = [self.target_qf1,self.target_qf2,self.target_qf3,self.target_qf4,self.target_qf5]    


        self.replay_buffer_online = EnvReplayBuffer(
        args.online_size,
        expl_env,
        )
        
        self.q1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.q2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)
        self.q3_optimizer = optimizer_class(self.qf3.parameters(), lr=qf_lr)
        self.q4_optimizer = optimizer_class(self.qf4.parameters(), lr=qf_lr)
        self.q5_optimizer = optimizer_class(self.qf5.parameters(), lr=qf_lr)

    # ensemble critic
    def compute_loss_ensembleq(self, data, online, gamma=0.99):
        obs, actions, rewards, next_obs, terminals = data['observations'], data[
            'actions'], data['rewards'], data['next_observations'], data['terminals']
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions) 
        q3_pred = self.qf3(obs, actions)
        q4_pred = self.qf4(obs, actions)
        q5_pred = self.qf5(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        M = np.random.choice(5, 2, replace=False)

        target_q_values = torch.min(
                self.target_qf_list[M[0]](next_obs, new_next_actions),
                self.target_qf_list[M[1]](next_obs, new_next_actions)
            ) # use the default conservative min random q 

        q_target = self.reward_scale * rewards + (1. - terminals) * gamma * target_q_values
        q_target = q_target.detach()

        qf1_loss = self.qf_criterion(q1_pred,q_target)
        qf2_loss = self.qf_criterion(q2_pred,q_target)
        qf3_loss = self.qf_criterion(q3_pred,q_target)
        qf4_loss = self.qf_criterion(q4_pred,q_target)
        qf5_loss = self.qf_criterion(q5_pred,q_target)

        if online:
            return qf1_loss, qf2_loss,qf3_loss,qf4_loss,qf5_loss

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
        q3_rand = self._get_tensor_values(
            obs, random_actions_tensor, network=self.qf3)
        q4_rand = self._get_tensor_values(
            obs, random_actions_tensor, network=self.qf4)
        q5_rand = self._get_tensor_values(
            obs, random_actions_tensor, network=self.qf5)


        q1_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, network=self.qf2)
        q3_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, network=self.qf3)
        q4_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, network=self.qf4)
        q5_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, network=self.qf5)


        q1_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, network=self.qf2)
        q3_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, network=self.qf3)
        q4_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, network=self.qf4)
        q5_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, network=self.qf5)

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
        cat_q3 = torch.cat(
            [q3_rand - random_density, q3_next_actions -
                new_log_pis.detach(), q3_curr_actions - curr_log_pis.detach()], 1
        )
        cat_q4 = torch.cat(
            [q4_rand - random_density, q4_next_actions -
                new_log_pis.detach(), q4_curr_actions - curr_log_pis.detach()], 1
        )
        cat_q5 = torch.cat(
            [q5_rand - random_density, q5_next_actions -
                new_log_pis.detach(), q5_curr_actions - curr_log_pis.detach()], 1
        )

        min_qf1_loss = torch.logsumexp(
            cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(
            cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf3_loss = torch.logsumexp(
            cat_q3 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf4_loss = torch.logsumexp(
            cat_q4 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf5_loss = torch.logsumexp(
            cat_q5 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
        min_qf3_loss = min_qf3_loss - q3_pred.mean() * self.min_q_weight
        min_qf4_loss = min_qf4_loss - q4_pred.mean() * self.min_q_weight
        min_qf5_loss = min_qf5_loss - q5_pred.mean() * self.min_q_weight

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss
        qf3_loss = qf3_loss + min_qf3_loss
        qf4_loss = qf4_loss + min_qf4_loss
        qf5_loss = qf5_loss + min_qf5_loss

        return qf1_loss, qf2_loss,qf3_loss,qf4_loss,qf5_loss

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
        q3_pi = self.qf3(o, pi)
        q4_pi = self.qf4(o, pi)
        q5_pi = self.qf5(o, pi)

        q_pi_list = [q1_pi,q2_pi,q3_pi,q4_pi,q5_pi]
        q_pi = torch.mean(torch.stack(q_pi_list), dim=0)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * log_pi - q_pi).mean()
        pi_info = {}

        return loss_pi, pi_info

    def compute_loss_q(self, data, gamma=0.99):
        obs, actions, rewards, next_obs, terminals = data['observations'], data[
            'actions'], data['rewards'], data['next_observations'], data['terminals']
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        # with torch.no_grad():
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
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
            self.replay_buffer_online.add_sample(o, a, r, d, o2,env_info={})
            self.replay_buffer.add_sample(o, a, r, d, o2,env_info={})
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                self.logger.store(EpRet=int(ep_ret), EpLen=ep_len)
                o, ep_ret, ep_len = self.explore_env.reset(), 0, 0 

  
    def train_offline(self, online_steps, train_steps, polyak=0.995, ue_loss_k=1000, ue_rollout=1000, ue_train_epoch=50, augment_mc='gain'):
        online = False
        for t in range(int(train_steps)):
            p_t = np.random.uniform()
            if p_t < self.args.p_online and self.replay_buffer_online.num_steps_can_sample() > 10000:
                np_data = self.replay_buffer_online.random_batch(self.args.batch_size)
                online = True
            else:
                np_data = self.replay_buffer.random_batch(self.args.batch_size)
                online = False
            
            data = np_to_pytorch_batch(np_data)
  
            loss_pi, pi_info = self.compute_loss_pi(data)
            q1_loss, q2_loss, q3_loss,q4_loss,q5_loss = self.compute_loss_ensembleq(data,online)

            self.q1_optimizer.zero_grad()
            q1_loss.backward(retain_graph=True)
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward(retain_graph=True)
            self.q2_optimizer.step()

            self.q3_optimizer.zero_grad()
            q3_loss.backward(retain_graph=True)
            self.q3_optimizer.step()

            self.q4_optimizer.zero_grad()
            q4_loss.backward(retain_graph=True)
            self.q4_optimizer.step()

            self.q5_optimizer.zero_grad()
            q5_loss.backward(retain_graph=True)
            self.q5_optimizer.step()

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

                for p, p_targ in zip(self.qf3.parameters(), self.target_qf3.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.qf4.parameters(), self.target_qf4.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.qf5.parameters(), self.target_qf5.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)


    def print_log(self,online_interaction,t,time_start):
        norm_score, test_score = self.test_policy(eval_episodes=5)
        self.logger.store(TestScore=int(test_score))
        self.logger.store(NormScore=int(norm_score))
        self.logger.log_tabular("EnvSteps",online_interaction)
        self.logger.log_tabular("TimeSteps", t)
        self.logger.log_tabular("TestScore", average_only=True)
        self.logger.log_tabular("NormScore",average_only=True)
        self.logger.log_tabular("Time", int(time.time()-time_start))
        self.logger.dump_tabular()
