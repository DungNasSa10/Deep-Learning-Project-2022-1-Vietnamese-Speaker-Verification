import torch
import torch.nn as nn
from learning.metrics import accuracy


class AMSoftmax(nn.Module):
    def __init__(self, n_out: int, n_classes: int, margin:float = 0.3, scale: int = 15):
        super(AMSoftmax, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = n_out
        self.W = torch.nn.Parameter(torch.randn(n_out, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)

        if label_view.is_cuda: 
            label_view = label_view.cpu()

        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)

        if x.is_cuda: 
            delt_costh = delt_costh.cuda()
            
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss    = self.ce(costh_m_s, label)
        prec1   = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
        
        return loss, prec1


def loss_init(n_out: int = 512, n_classes: int = 1015, margin=0.3, scale=15, **kwargs):
    return AMSoftmax(n_out=n_out, n_classes=n_classes, margin=margin, scale=scale)
