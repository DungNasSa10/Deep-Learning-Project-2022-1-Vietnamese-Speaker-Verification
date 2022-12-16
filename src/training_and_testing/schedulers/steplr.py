from torch.optim.lr_scheduler import StepLR
from typing import Tuple


def scheduler(optimizer, lr_decay: float = 0.95, lr_step: int = 1, **kwargs) -> Tuple[StepLR, int]:
    """
    StepLR scheduler based on Pytorch.

    Args
    -----
        optimizer (torch.optim.optimizer.Optimizer): Wrapped optimizer.
        lr_step (int): Period of learning rate decay. Default: 1.
        lr_decay (float): Multiplicative factor of learning rate decay. Default: 0.95.

    Return
    -----
        scheduler (torch.optim.lr_scheduler.StepLR): learning rate scheduler
        lr_step (int): learning rate decay step type
    """
    scheduler = StepLR(optimizer, step_size = lr_step, gamma = lr_decay)
    lr_step_type = "epoch"

    print('Initialised StepLR scheduler')

    return scheduler, lr_step_type