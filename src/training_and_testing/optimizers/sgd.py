from torch.optim import SGD


def optimizer(params, lr: float = 1e-3, weight_decay: float = 2e-5, **kwargs) -> None:
    """
    Adam optimizer based on Pytorch.

    Args
    -----
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 2e-5)

    Return
    -----
        torch.optim.SGD
    """
    print("Initialised SGD optimizer")

    return SGD(params = params, lr = lr, weight_decay = weight_decay)
