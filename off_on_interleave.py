import gym
import numpy as np
import time,os
import torch
from utils.logx import EpochLogger
import yaml


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="walker2d-medium-v0")  # OpenAI gym environment name (need to be consistent with the dataset name)
    parser.add_argument("--plot_name", type=str, default="debug")  # OpenAI gym environment name (need to be consistent with the dataset name)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--w", default=1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)  # Mini batch size for networks
    parser.add_argument("--alpha", default=2.5,type=float)  # Target network update rate [1,4]
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e3, type=int)  # 1e4 How often (time steps) we evaluate the policy
    parser.add_argument("--print_freq", default=3e3, type=int)  # 1e4 How often (time steps) we print out log
    parser.add_argument("--online_size", default=20000, type=int)  # online replay buffer size
    parser.add_argument("--max_envsteps", default=100000, type=int)
    parser.add_argument("--max_vae_trainstep", default=2e5, type=int) #2e5
    parser.add_argument('--engine', type=str, default="P0")
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--p_online', type=float, default=0.5)
    parser.add_argument("--T_i", default=100000, type=int)
    parser.add_argument("--T_o", default=10000, type=int)
    parser.add_argument("--explore_steps", default=1000, type=int)

    # Need change every time before running
    parser.add_argument('--version', type=str, default="")
    parser.add_argument('--offline', dest="offline",  action='store_true') # defult False
    parser.add_argument('--load_model', dest="load_model",  action='store_true') # defult False
    parser.add_argument('--save_model', dest="save_model",  action='store_true') # defult False

    parser.add_argument('--itr', default=None,  type=int) # model's name
    parser.add_argument('--model_dir', default="",  type=str) # model's name

    # BCQ parameter
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.1, type=float)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--load_buffer_size", default=1000000, type=int)  # number of samples to load into the buffer
    parser.add_argument("--actor_lr", default=1e-3, type=float) # learning rate of actor
    parser.add_argument("--n_action", default=100, type=int) # number of sampling action for policy (in backup)
    parser.add_argument("--n_action_execute", default=100, type=int) # number of sampling action for policy (in execution)

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    max_performance = -np.inf

    # Get the settings from exp.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "exp.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    

    if "debug" in args.version:
        offline_training_steps = 100
        online_explore_steps = 100
        max_steps = 20000 #200
        init_t = 10
        args.batch_size = 3
        test_steps = 100
        datadir = "./data/debug/"
        args.average_window = 100
    else:
        max_steps = args.max_envsteps
        init_t = args.T_i 
        test_steps = 5000
        online_explore_steps = args.explore_steps # should > 1000
        args.version = config_dict["version"]
        datadir = config_dict["data_dir"]
        offline_training_steps = config_dict["offline_training_steps"]

    args.engine = config_dict["engine"]
    args.plot_name = config_dict["plot_name"]
    args.exp_name = config_dict["exp_name"]

    if args.plot_name == "td3_on":
        init_t = 0

    if args.save_model and args.load_model: assert False
    if args.load_model: assert args.itr != None; args.model_dir != ""

    from utils.logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args,args.version,args.env,args.exp_name,args.plot_name, args.seed,args.model_dir,data_dir=datadir)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(vars(args))
    print("logger_kwargs",logger_kwargs)


    exec("from agents.{} import agent".format(args.engine))

    agent = agent(args,logger)

    '''
    offline-online interleave learning
    '''
    time_start = time.time()
    t_off = 0
    online_interaction = 0

    agent.test_policy(eval_episodes=5)

    while t_off < int(init_t):
        'offline pretraining'
        agent.train_offline(0,test_steps)
        t_off += test_steps
        agent.print_log(0,t_off,time_start)


    while online_interaction <= int(max_steps):
        'online explore'
        # print("online exploration")
        agent.explore(online_explore_steps)
        online_interaction += online_explore_steps

        'offline learning'
        # print("offline learning")
        agent.train_offline(online_explore_steps,offline_training_steps)
        t_off += offline_training_steps

        agent.print_log(online_interaction,t_off,time_start)
