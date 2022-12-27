from typing import Tuple
import torch
import torch.nn as nn
from learning.metrics import accuracy


class SoftmaxLoss(nn.Module):
    def __init__(self, n_out: int = 512, n_classes: int = 1015) -> None:
        super().__init__()

        self.test_normalize = True
        self.loss_func = nn.CrossEntropyLoss()
        self.fc = nn.Linear(n_out, n_classes)

    def forward(self, x: torch.Tensor, label: torch.Tensor=None) -> Tuple[torch.Tensor]:
        """
        Args
        ----
            x (Tensor):
            label (Tensor, optional)

        Return
        ------
            tuple of torch.Tensor and torch.Tensor
            loss and precision 1 
        """
        x = self.fc(x)
        loss = self.loss_func(x, label)
        prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


def loss_init(n_out: int = 512, n_classes: int = 1015, **kwargs):
    return SoftmaxLoss(n_out=n_out, n_classes=n_classes)