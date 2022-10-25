import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


LOG_STD_MAX = 2
LOG_STD_MIN = -20



class ReplayBuffer:
    """
    A simple FIFO experience replay buffer adapted from SAC agents. 
    add the discounted return calculation.
    """

    def __init__(self, obs_dim, act_dim, size, logger,gamma=0.99):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.not_done_buf = np.zeros(size, dtype=np.float32)
        self.discounted_returns = np.array([])
        self.early_cut = np.array([])
        self.terminate_states = np.array([])
        self.gamma = gamma 
        self.logger = logger
        self.trajectories = {}
        self.trajectories_discounted_return = np.array([])

        self.ptr, self.size, self.max_size = 0, 0, size

        # self.states_idx = np.array([])

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
        self.logger.save_print(f"buffer size:{self.size}")
        # self.calucate_discounted_return()
        # self.states_idx = np.array([i for i in range(self.size)])
        # self.states_idx_all = self.states_idx

    def get_trajectory(self, start_idxs):
        states_idx = []

        for idx in start_idxs:
            end = self.terminate_states[idx].item()+1  # include the end idx
            states_idx.extend([i for i in range(idx, int(end))])

        return states_idx

    def calucate_discounted_return(self):
        # iterate start from self.size-1 to 0, state is represented by its index
        discounted_return = 0
        undiscounted_return = 0
        prev_s = None
        discounted_returns = np.array([])
        undiscounted_returns = np.array([])
        trajectory = np.array([],dtype=int)
        trajectories = {}
        trajectories_discounted_return = np.array([])

        early_cut = np.array([])
        terminate_states = np.array([], dtype=int)
        terminate_state = None
        trajectory_lenght = 0
        trajectories_idx = 0

        for ind in range(self.size-1, -1, -1):
            done = 1-self.not_done_buf[ind]
            state = self.obs_buf[ind]
            next_state = self.obs2_buf[ind]
            reward = self.rew_buf[ind]

            if done:  # normally terminted state == next_state !!
                if trajectory.size != 0:
                    self.trajectories[trajectories_idx] = trajectory[::-1]
                    trajectories_idx += 1
                    # if trajectories.size == 0: 
                    #     trajectories = trajectory[::-1].reshape(1,)
                    # else:
                    #     trajectories = np.stack(trajectories,trajectory[::-1])
                    trajectories_discounted_return = np.append(trajectories_discounted_return,discounted_return)
                    trajectory = np.array([])
                # else:
                trajectory = np.append(trajectory,ind)
                
                trajectory_lenght = 1
                discounted_return = reward
                undiscounted_return = reward
                discounted_returns = np.append(
                    discounted_returns, discounted_return)
                undiscounted_returns = np.append(
                    undiscounted_returns, undiscounted_return)
                terminate_state = ind
                prev_s = state
                early_cut = np.append(early_cut, 0)
                terminate_states = np.append(terminate_states, terminate_state)
                continue

            if np.array_equal(prev_s, next_state):  # normal mid state
                discounted_return = self.gamma*discounted_return + reward
                undiscounted_return = 1*undiscounted_return + reward
                trajectory = np.append(trajectory,ind)
                prev_s = state
                early_cut = np.append(early_cut, 0)
                trajectory_lenght += 1
            else:  # early cutoff
                if trajectory.size != 0:
                    self.trajectories[trajectories_idx] = trajectory[::-1]
                    trajectories_idx += 1
                    trajectories_discounted_return = np.append(trajectories_discounted_return,discounted_return)
                    trajectory = np.array([])
                # else:
                trajectory = np.append(trajectory,ind)

                trajectory_lenght = 1
                discounted_return = reward
                undiscounted_return = reward
                terminate_state = ind
                prev_s = state
                early_cut = np.append(early_cut, 1)


            discounted_returns = np.append(
                discounted_returns, discounted_return)
            undiscounted_returns = np.append(
                undiscounted_returns, undiscounted_return)
            terminate_states = np.append(terminate_states, terminate_state)
        
        self.logger.save_print(f"buffer size:{self.size}")
        self.logger.save_print(
            f"mean distounded_returns:{np.mean(discounted_returns):.3f},mean undistounded_returns:{np.mean(undiscounted_returns):.3f}")
        self.logger.save_print(
            f"median distounded_returns:{np.median(discounted_returns):.3f},median undistounded_returns:{np.median(undiscounted_returns):.3f}")
        self.logger.save_print(
            f"std distounded_returns:{np.std(discounted_returns):.3f},std undistounded_returns:{np.std(undiscounted_returns):.3f}")
        self.logger.save_print(
            f"max distounded_returns:{np.max(discounted_returns):.3f},max undistounded_returns:{np.max(undiscounted_returns):.3f}")
        self.logger.save_print(
            f"min distounded_returns:{np.min(discounted_returns):.3f},min undistounded_returns:{np.min(undiscounted_returns):.3f}")
        
        # self.logger.save_print(f"percentage of undiscounted returns > 300:{np.sum(undiscounted_returns>=300)/len(undiscounted_returns):.3f}")
        # self.logger.save_print(f"percentage of undiscounted returns > 500:{np.sum(undiscounted_returns>=500)/len(undiscounted_returns):.3f}")
        # self.logger.save_print(f"percentage of undiscounted returns > 800:{np.sum(undiscounted_returns>=800)/len(undiscounted_returns):.3f}")
        self.logger.save_print(f"percentage of undiscounted returns < 1000:{np.sum(undiscounted_returns<1000)/len(undiscounted_returns):.3f}")
        # self.logger.save_print(f"percentage of undiscounted returns > 1500:{np.sum(undiscounted_returns>=2000)/len(undiscounted_returns):.3f}")
        self.logger.save_print(f"percentage of undiscounted returns > 3000:{np.sum(undiscounted_returns>3000)/len(undiscounted_returns):.3f}")

        self.trajectories_discounted_return = trajectories_discounted_return
        self.max_discounted_return = np.max(discounted_returns)
        # assert max(early_cut) == 0
        # print(f"self.trajectories[0]:{self.trajectories[0]},self.trajectories[1]:{self.trajectories[1]}")
        self.discounted_returns,self.early_cut,self.terminate_states = discounted_returns[::-1], early_cut[::-1], terminate_states[::-1]



    def calcuate_n_step_return(self,ind_start):
        discounted_return = self.discounted_returns[ind_start].reshape(-1)
        terminal_states_idx = self.terminate_states[ind_start]

        early_cut = self.early_cut[terminal_states_idx]
        early_cut_state_idx = np.where(early_cut == 1)
        is_empty = early_cut_state_idx[0].size == 0

        if is_empty:
            return discounted_return,is_empty,early_cut_state_idx

        assert False

        early_cut_state_idx = terminal_states_idx[early_cut_state_idx]
        terminal_states_unique = np.unique(terminal_states_idx)

        terminal_states = torch.FloatTensor(
            self.obs2_buf[terminal_states_unique]).to(device)

        # *torch.FloatTensor(early_cut)#.to(device)
        v_last = self.approx_v(terminal_states)

        for state_idx, v in zip(terminal_states_unique, v_last):
            v_last_idx = np.where(terminal_states_idx == state_idx)
            # v need to multiple gamma's N
            discounted_return[v_last_idx] += v.item()*self.gamma

        return discounted_return,is_empty,early_cut_state_idx


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.obs_buf[idxs],
                     next_state=self.obs2_buf[idxs],
                     action=self.act_buf[idxs],
                     reward=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     not_done=self.not_done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}



class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='relu', init_small_weights=False, init_w=1e-3):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        if init_small_weights:
            for affine in self.affine_layers:
                affine.weight.data.uniform_(-init_w, init_w)
                affine.bias.data.uniform_(-init_w, init_w)


    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim) # full connected layer
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim) # full connected layer
        self.act_limit = act_limit

    def forward(self, obs, act=None, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            if act is not None: 
                pi_action = act

            logp_pi = pi_distribution.log_prob(pi_action).sum(-1)#axis=
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(1) #axis=
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # self.cude()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorNCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),activation=nn.ReLU):
        super().__init__()
        # self.cuda()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)

        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        # More
        self.q3 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q4 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q5 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def q(self,state,action):

        q1 = self.q1(state,action)
        q2 = self.q2(state,action)
        # More
        q3 = self.q3(state,action)
        q4 = self.q4(state,action)
        q5 = self.q5(state,action)


        return [q1,q2,q3,q4,q5]#,q6,q7,q8,q9,q10]


    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
            return a.cpu().numpy() # a.numpy()

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
            return a.cpu().numpy()



class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPActorCriticTd3(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

class MLPActorNCriticTd3(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

        # More
        self.q3 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q4 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q5 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def q(self,state,action):
        q1 = self.q1(state,action)
        q2 = self.q2(state,action)
        # More
        q3 = self.q3(state,action)
        q4 = self.q4(state,action)
        q5 = self.q5(state,action)

        return [q1,q2,q3,q4,q5]

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()


class MLPActorCritic_ABM(nn.Module):
# adapted from sac
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        # self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        # self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
            return a


class Actor_VPG(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPGaussianActor_VPG(Actor_VPG):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution



class MLPActorCritic_VPG(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        # if isinstance(action_space, Box):
        self.pi = MLPGaussianActor_VPG(obs_dim, action_space.shape[0], hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]