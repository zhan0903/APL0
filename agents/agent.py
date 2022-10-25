import time
from torch.utils.tensorboard import SummaryWriter

# base class for other agent to inherient, interface class
class agent(object):
    def __init__(self,args,logger) -> None:
        self.logger = logger
        work_dir = f"./tensorboard_curves/{args.exp_name}/{args.env}/{args.plot_name}/{args.version}/s_{args.seed}"
        self.writer = SummaryWriter(work_dir)

    def save_models(self):
        raise NotImplementedError

    def load_models(self):
        raise NotImplementedError
    