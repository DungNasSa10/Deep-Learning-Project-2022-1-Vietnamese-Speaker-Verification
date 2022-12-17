import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tuneThreshold import tuneThresholdfromScore
import random

class LossFunction(nn.Module):

    def __init__(self, hard_rank=0, hard_prob=0, margin=0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.hard_rank  = hard_rank
        self.hard_prob  = hard_prob
        self.margin     = margin

        print('Initialised Triplet Loss')

    def forward(self, x, label=None):
        # The input tensor x is expected to have shape 
        # (batch_size, 2, feature_dim)
        assert x.size()[1] == 2
        

        out_anchor      = F.normalize(x[:,0,:], p=2, dim=1)
        out_positive    = F.normalize(x[:,1,:], p=2, dim=1)
        stepsize        = out_anchor.size()[0]

        # calculate the squared distance
        output      = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1),out_positive.unsqueeze(-1).transpose(0,2))**2)

        negidx      = self.mineHardNegative(output.detach())

        out_negative = out_positive[negidx,:]

        labelnp     = np.array([1]*len(out_positive)+[0]*len(out_negative))

        ## calculate distances
        pos_dist    = F.pairwise_distance(out_anchor,out_positive)
        neg_dist    = F.pairwise_distance(out_anchor,out_negative)

        ## loss function
        nloss   = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))

        scores = -1 * torch.cat([pos_dist,neg_dist],dim=0).detach().cpu().numpy()

        errors = tuneThresholdfromScore(scores, labelnp, []);

        return nloss, errors[1]

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Hard negative mining
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def mineHardNegative(self, output):
        """
        negatives are mined by sorting the 
        similarities between the anchor and positive 
        samples in descending order and selecting 
        the top ranked.

        Args
        ----
            output: similarities between the anchor 
            and positive samples

        Return
        ----
            negidx: negative samples
        """

        negidx = []

        for idx, similarity in enumerate(output):

            simval, simidx = torch.sort(similarity,descending=True)

            # control the number of negative samples by hard_rank
            # control the probability of negative samples by hard_prob
            if self.hard_rank < 0:

                ## Semi hard negative mining

                semihardidx = simidx[(similarity[idx] - self.margin < simval) &  (simval < similarity[idx])]

                if len(semihardidx) == 0:
                    negidx.append(random.choice(simidx))
                else:
                    negidx.append(random.choice(semihardidx))

            else:

                ## Rank based negative mining
                
                simidx = simidx[simidx!=idx]

                if random.random() < self.hard_prob:
                    negidx.append(simidx[random.randint(0, self.hard_rank)])
                else:
                    negidx.append(random.choice(simidx))

        return negidx

if __name__ == '__main__':
    # create instance of LossFunction
    loss_fn = LossFunction()

    # define test input tensor x with shape (batch_size, 2, feature_dim)
    x = torch.Tensor([[[-1.,  1.],
                       [ 1.,  0.]],

                      [[-1.,  2.],
                       [-1., -2.]]])

    # define test label
    label = None

    # call forward method on test input
    output, error = loss_fn.forward(x, label)

    # define expected output values
    expected_output = 0.8385
    expected_error = 0.0

    # assert output values are as expected
    assert output == expected_output, f'Error: expected {expected_output} but got {output}'
    assert error == expected_error, f'Error: expected {expected_error} but got {error}'   