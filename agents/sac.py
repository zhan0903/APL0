from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from agents import core,agent
from utils.logx import EpochLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.not_done_buf = np.zeros(size, dtype=np.float32)
        self.discounted_returns = np.array([])
        self.early_cut = np.array([])
        self.terminate_states = np.array([])

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def load(self, save_folder, size=-1):
        reward = [];not_done = []
        reward_buffer = np.load(f"{save_folder}_reward.npy")
        not_done_buffer = np.load(f"{save_folder}_not_done.npy")
        
        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.obs_buf[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.act_buf[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.obs2_buf[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        for r,n_d in zip(reward_buffer,not_done_buffer):
            reward.append(float(r[0]))
            not_done.append(int(n_d[0]))
        self.rew_buf[:self.size] = reward[:self.size]
        self.not_done_buf[:self.size] = not_done[:self.size]
        self.done_buf[:self.size] = 1-self.not_done_buf[:self.size]

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.obs_buf[idxs],
                     next_state=self.obs2_buf[idxs],
                     action=self.act_buf[idxs],
                     reward=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     not_done=self.not_done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}

class agent(agent.agent):
    def __init__(self,args,logger,lr=1e-3,actor_critic=core.MLPActorCritic,ac_kwargs=dict(),replay_size=int(1e6)):
        super().__init__(args,logger)
        print("debug in sac agent")
        self.args = args
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.logger = logger
        self.all_steps = 0
        self.alpha=0.2 #default 0.2

        self.env = gym.make(args.env)
        self.env.seed(args.seed)
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        if self.args.offline:
            self.replay_buffer.load(f"./buffers/{args.dataset}", args.load_buffer_size)


    def get_action(self,o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), 
                      deterministic)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data,gamma=0.99):
        o, a, r, o2, n_d = data['state'], data['action'], data['reward'], data['next_state'], data['not_done']
        d = 1-n_d

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        # assert False
        o = data['state']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
        self.logger.store(logp_pi_t=logp_pi.mean().item(),
                        q_pi_t=q_pi.mean().item())

        return loss_pi, pi_info

    def update(self,data,polyak=0.995):
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        if self.args.version != "debug":
            self.writer.add_scalar("pi_loss", loss_pi.item(), self.all_steps)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def test_policy(self, eval_episodes=5):
        # self.policy.cpu()
        eval_env = gym.make(self.args.env)
        eval_env.seed(self.args.seed + 100)
        avg_reward = 0.
        trajectory_lenght = 0
        
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.get_action(np.array(state),True)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
                trajectory_lenght += 1

        avg_reward /= eval_episodes
        trajectory_lenght /= eval_episodes

        return avg_reward,trajectory_lenght

    def train(self, batch_size=100):
        time_start = time.time()
        for t in range(int(self.args.max_timesteps)):
            batch = self.replay_buffer.sample(batch_size)
            self.update(data=batch)
            self.all_steps += 1

            if t % self.args.eval_freq == 0: # eval_freq should be 1000
                # assert self.args.eval_freq == 1000
                test_score,trajectory_lenght = self.test_policy(eval_episodes=5)
                self.logger.store(TestScore=int(test_score))

            if t % self.args.print_freq == 0: # print every 3000 steps
                # assert self.args.print_freq == 3000
                if not ("debug" in self.args.version):
                    self.writer.add_scalar("q_pi_t",self.logger.get_stats("q_pi_t")[0],t)
                    self.writer.add_scalar("logp_pi_t",self.logger.get_stats("logp_pi_t")[0],t)
                    self.writer.add_scalar("TestScore",self.logger.get_stats("TestScore")[0],t)
                    
                self.logger.log_tabular("TimeSteps",t)
                self.logger.log_tabular("TestScore",average_only=True)
                self.logger.log_tabular("trajectory_lenght",trajectory_lenght)
                self.logger.log_tabular("Time",int(time.time()-time_start))
                self.logger.dump_tabular()

        self.writer.flush()
        self.writer.close()

    def train_sac_online(self,start_steps=10000,batch_size=100,max_ep_len=1000,update_after=1000, update_every=50):#update_every=50
        # Prepare for interaction with environment
        # total_steps = steps_per_epoch * epochs
        time_start = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for t in range(int(self.args.max_timesteps)):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if self.all_steps > start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            self.all_steps += 1

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if self.all_steps >= update_after and self.all_steps % update_every == 0:
                for j in range(update_every):
                    batch = self.replay_buffer.sample(batch_size)
                    self.update(data=batch)

            if t % self.args.print_freq == 0: # print every 3000 steps

                test_score,trajectory_lenght = self.test_policy(eval_episodes=5)

                self.logger.log_tabular("TimeSteps",t)
                self.logger.log_tabular("TestScore",test_score)
                self.logger.log_tabular("Time",int(time.time()-time_start))
                self.logger.log_tabular("trajectory_lenght",trajectory_lenght)
                self.logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    # (version, env, engine, seed=None, data_dir=None, datestamp=False):

    from utils.logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs("sac_2021_0111_2139", args.env, args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

