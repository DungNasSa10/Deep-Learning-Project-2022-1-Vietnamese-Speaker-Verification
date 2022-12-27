import torch.nn as nn
from .softmax import SoftmaxLoss
from .angleproto import AngleProto


class SoftmaxProto(nn.Module):
    """
    Softmaxproto combines the functionality 
    of two other loss functions: 
    softmax.SoftmaxProto and angleproto.SoftmaxProto.
    """

    def __init__(self, n_out: int = 512, n_classes: int = 1015, init_w=10.0, init_b=-5.0):
        super(SoftmaxProto, self).__init__()

        self.test_normalize = True

        self.softmax = SoftmaxLoss(n_out=n_out, n_classes=n_classes)
        self.angleproto = AngleProto(init_w=init_w,  init_b=init_b)

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):
        # The input tensor x is expected to have shape 
        # (batch_size, 2, feature_dim)
        assert x.size()[1] == 2

        nlossS, prec1   = self.softmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))

        nlossP, _       = self.angleproto(x,None)

        return nlossS+nlossP, prec1


def loss_init(n_out: int = 512, n_classes: int = 1015, init_w=10.0, init_b=-5.0, **kwargs):
    return SoftmaxProto(n_out=n_out, n_classes=n_classes, init_w=init_w, init_b=init_b)