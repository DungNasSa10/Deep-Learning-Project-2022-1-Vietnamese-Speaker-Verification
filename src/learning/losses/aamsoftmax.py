import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from learning.metrics import accuracy


class AAMSoftmax(nn.Module):
    def __init__(self, n_out: int, n_classes: int, margin: float = 0.2, scale: int = 30, easy_margin: bool = False) -> None:
        super(AAMSoftmax, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = n_out
        self.w = torch.nn.Parameter(torch.FloatTensor(n_classes, n_out), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.w, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialized AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None) -> None:

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.w))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


def loss_init(n_out: int = 512, n_classes: int = 1015, margin: float = 0.2, scale:int = 30, easy_margin: bool = False, **kwargs):
    return AAMSoftmax(n_out=n_out, n_classes=n_classes, margin=margin, scale=scale, easy_margin=easy_margin)