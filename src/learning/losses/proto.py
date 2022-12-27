import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from learning.metrics import accuracy


class Proto(nn.Module):
    def __init__(self):
        super().__init__()
        self.test_normalize = False
        self.criterion  = torch.nn.CrossEntropyLoss()
        print('Initialised Prototypical Loss')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        output  = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))**2)
        label   = torch.from_numpy(np.asarray(range(0,stepsize))).to('cuda' if torch.cuda.is_available() else 'cpu')
        nloss   = self.criterion(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1


def loss_function(**kwargs):
    return Proto()