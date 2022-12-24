from torch.optim.lr_scheduler import CyclicLR
from typing import Tuple


def scheduler_init(optimizer, base_lr: float = 1e-6, max_lr: float = 1e-3, step_size_up: int = 5000, step_size_down: int = 10000, mode: str = "triangular2", **kwargs) -> Tuple[CyclicLR, int]:
    """
    CyclicLR scheduler based on Pytorch.

    Args
    -----
        optimizer (torch.optim.optimizer.Optimizer): Wrapped optimizer.
        base_lr (float, optional): Initial learning rate which is the lower boundary in the cycle for each parameter group. Default: 1e-6
        max_lr (float, optional): Upper learning rate boundaries in the cycle for each parameter group. Default: 1e-3
        step_size_up (int): Number of training iterations in the increasing half of a cycle. Default: 5000
        step_size_down (int): Number of training iterations in the decreasing half of a cycle. If step_size_down is None, it is set to step_size_up. Default: 10000
        mode (str): One of {triangular, triangular2, exp_range}. Values correspond to policies detailed above. If scale_fn is not None, this argument is ignored. Default: "triangular"

    Return
    -----
        scheduler (torch.optim.lr_scheduler.CyclicLR): learning rate scheduler
        lr_step (int): learning rate decay step type
    """
    
    scheduler = CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, step_size_up = step_size_up, 
                        step_size_down = step_size_down, mode = mode, cycle_momentum=False)
    lr_step_type = "iteration"

    print('Initialised CyclicLR scheduler')

    return scheduler, lr_step_type