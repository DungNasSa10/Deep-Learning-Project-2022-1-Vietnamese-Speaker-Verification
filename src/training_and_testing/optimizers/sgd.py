from torch.optim import SGD


def optimizer(params, lr: float = 1e-3, weight_decay: float = 2e-5, **kwargs) -> SGD:
    """
    Adam optimizer based on Pytorch.

    Args
    ----
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Default: 1e-3.
        weight_decay (float, optional): weight decay (L2 penalty). Default: 2e-5.

    Returns
    -------
        optimizer: torch.optim.SGD
    """
    print("Initialised SGD optimizer")

    optimizer = SGD(params = params, lr = lr, weight_decay = weight_decay)

    return optimizer
