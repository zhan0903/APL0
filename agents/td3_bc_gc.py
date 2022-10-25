import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import d4rl
import numpy as np
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(3e6)):
        self.max_size = max_size
        self.ptr = 0
        self._size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward#.reshape(-1,1)
        self.not_done[self.ptr] = 1. - done#.reshape(-1,1)

        self.ptr = (self.ptr + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def num_steps_can_sample(self):
        return self._size

    def sample(self, batch_size):
        ind = np.random.randint(0, self._size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )


    def convert_D4RL(self, dataset):
        self._size = dataset['observations'].shape[0]
        self.state[:self._size] = dataset['observations']
        self.action[:self._size] = dataset['actions']
        self.next_state[:self._size] = dataset['next_observations']
        self.reward[:self._size] = dataset['rewards'].reshape(-1,1)
        self.not_done[:self._size] = 1. - dataset['terminals'].reshape(-1,1)
        # self.size = self.state.shape[0]
        self.ptr = (self.ptr + self._size) % self.max_size


    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class agent(object):
    def __init__(
        self,
        args,
        logger,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):
        self.explore_env = gym.make(args.env)
        self.explore_env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.max_action = self.explore_env.action_space.high[0]
        self.action_dim = self.explore_env.action_space.shape[0] 
        self.state_dim = self.explore_env.observation_space.shape[0]
        self.env_name = args.env
        self.seed = args.seed
        self.logger = logger

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise*self.max_action
        self.noise_clip = noise_clip*self.max_action
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.p_online = args.p_online

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.explore_env))

        self.replay_buffer_online = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(args.online_size))

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def explore_action(self,o, expl_noise=0.1):
        a = self.select_action(np.array(o))
        a += np.random.normal(0, self.max_action * expl_noise, size=self.action_dim)
        return np.clip(a, -self.max_action, self.max_action)

    def test_policy(self, seed_offset=100, eval_episodes=5):
        eval_env = gym.make(self.env_name)
        eval_env.seed(self.seed + seed_offset)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                # state = (np.array(state).reshape(1,-1) - mean)/std
                action = self.select_action(state)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes
        d4rl_score = 100*eval_env.get_normalized_score(avg_reward)

        # print("---------------------------------------")
        # print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
        # print("---------------------------------------")
        return d4rl_score,avg_reward


    def explore(self,online_explore_steps,max_ep_len=1000):
        o, ep_ret, ep_len = self.explore_env.reset(), 0, 0
        for _ in range(online_explore_steps):
            a = self.explore_action(o)
            o2, r, d, _ = self.explore_env.step(a)               
            ep_ret += r
            ep_len += 1
            d = False if ep_len == max_ep_len else d
            # state, action, next_state, reward, done
            self.replay_buffer_online.add(o, a, o2, r, d)
            self.replay_buffer.add(o, a, o2, r, d)
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                self.logger.store(EpRet=int(ep_ret), EpLen=ep_len)
                o, ep_ret, ep_len = self.explore_env.reset(), 0, 0 


    def train_offline(self, explore_steps,train_steps, batch_size=256):
        w = 0

        for t in range(int(train_steps)):
            self.total_it += 1
            p_t = np.random.uniform()
            if p_t < self.p_online and self.replay_buffer_online.num_steps_can_sample() > 10000:
                state, action, next_state, reward, not_done = self.replay_buffer_online.sample(batch_size)
                w = 0
            else:
                state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)
                w = 1

        # Sample replay buffer 
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) +  F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor loss
                pi = self.actor(state)
                Q = self.critic.Q1(state, pi)
                lmbda = self.alpha/Q.abs().mean().detach()

                actor_loss = -lmbda * Q.mean() + w * F.mse_loss(pi, action) 
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def print_log(self,online_interaction,t,time_start):

        norm_score, test_score = self.test_policy(eval_episodes=5)
        self.logger.store(NormScore=int(norm_score))
        self.logger.log_tabular("EnvSteps",online_interaction)
        self.logger.log_tabular("TimeSteps", t)
        self.logger.log_tabular("NormScore",average_only=True)
        self.logger.log_tabular("Time", int(time.time()-time_start))
        self.logger.log_tabular("ReplayBufferS",self.replay_buffer_online._size)
        self.logger.log_tabular("ReplayBuffer",self.replay_buffer._size)
        self.logger.dump_tabular()


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)