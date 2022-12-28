import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):

        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):

    def __init__(self, channel, reduction=8):

        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class PreEmphasis(nn.Module):

    def __init__(self, coef: float = 0.97, squeeze_output: bool=False):
        """
        This layer is based on: 
        https://www.researchgate.net/publication/327594319_A_Complete_End-to-End_Speaker_Verification_System_Using_Deep_Neural_Networks_From_Raw_Signals_to_Verification_Result

        Args
        ----
            coef (float): coefficient of filter
        """
        super().__init__()
        flipped_filter = torch.FloatTensor([-coef, 1.0]).unsqueeze(0).unsqueeze(0)
        self.squeeze_output = squeeze_output
        self.register_buffer("flipped_filter", flipped_filter)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
            input (Tensor): shape is (1, T)
        """
        assert len(input.size()) == 2, f"Expect input has 2 dimensions, got {len(input.size())}"

        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), mode='reflect')
        output = F.conv1d(input, self.flipped_filter)
        
        if self.squeeze_output:
            return output.squeeze(1)
        return output


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):

        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):

        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)

        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)

        return x


class AFMS(nn.Module):
    """
    Rawnet2 and FMS: https://arxiv.org/abs/2004.00526
    """
    def __init__(self, nb_dim: int) -> None:
        """
        Args
        ----
            nb_dim (int)
        """
        super().__init__()

        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.adaptive_avg_pool1d(x, output_size=1)
        y = y.view(x.size(0), -1)
        y = self.sigmoid(self.fc(y))
        y = y.view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y

        return x


class Res2MaxPoolingBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int=None,
        dilation: int=None,
        scale: int = 4,
        max_pooling_size: int=None
    ) -> None:
        """
        Args
        ----
            in_channels (int):      input channels of input tensor
            out_channels (int):     number of channels of output tensor after getting through `forward` 
            kernel_size (int):      kernel sizes of Res2Dilated convolutional block
            dilation (int):         dilation of Res2Dilated convolutional block
            scale (int):
            max_pooling_size (int): kernel_size of max pooling layer
        """
        super().__init__()

        self.width = int(math.floor(out_channels / scale))

        ### First Conv1d + BN
        self.conv_1 = nn.Conv1d(in_channels, self.width * scale, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(self.width * scale)

        
        self.nums = scale - 1
        num_pad = math.floor(kernel_size / 2) * dilation
        
        ### Res2Dilated Conv1d + BN
        self.res2dilated_convs = nn.ModuleList([
            nn.Conv1d(
                self.width,
                self.width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=num_pad
            ) for _ in range(self.nums)
        ])

        self.res2dilated_bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

        ### Final Conv1d + BN
        self.conv_3 = nn.Conv1d(self.width * scale, out_channels, kernel_size=1)
        self.bn_3 = nn.BatchNorm1d(out_channels)

        ### ReLU + max pooling + AFMS
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(max_pooling_size) if max_pooling_size else None
        self.afms = AFMS(out_channels)

        ### Skip connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
                if in_channels != out_channels
                else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ### First Conv1d + ReLU + BN
        output = self.conv_1(x)
        output = self.relu(output)
        output = self.bn_1(output)

        ### Res2Dilated Conv1d + ReLU + BN
        splits = torch.split(output, self.width, dim=1)
        for i in range(self.nums):
            sp = splits[0] if i == 0 else sp + splits[i]
            sp = self.res2dilated_convs[i](sp)
            sp = self.relu(sp)
            sp = self.res2dilated_bns[i](sp)

            output = sp if i == 0 else torch.cat([output, sp], dim=1)
        
        output = torch.cat([output, splits[self.nums]], dim=1)

        ### Final Conv1d + ReLu + BN
        output = self.conv_3(output)
        output = self.relu(output)
        output = self.bn_3(output)

        ### Skip connection
        output += self.residual(x)

        ### Max pooling
        if self.max_pooling is not None:
            output = self.max_pooling(output)
        
        ### AFMS
        output = self.afms(output)

        return output