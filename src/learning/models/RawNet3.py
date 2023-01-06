import torch
import torch.nn as nn

from asteroid_filterbanks import Encoder, ParamSincFB
from .layers import PreEmphasis, Res2MaxPoolingBlock


class RawNet3(nn.Module):
    def __init__(
        self,
        model_scale: int,
        context: bool,
        summed: bool,
        channels: int=1024,
        encoder_type: str='ECA',
        log_sinc: bool=None,
        norm_sinc: bool='mean',
        sinc_stride: int=None,
        out_bn: bool=None,
        n_out: int=None,
        **kwargs
    ) -> None:
        r"""
        RawNet3 paper: https://arxiv.org/abs/2203.08488

        Args
        ----
            model_scale (int): 
                scale of Res2MaxPooling blocks
            context (bool): 
                use context in attention layers
            summed (bool): 

            channels (int): 
                number of hidden channels
            encoder_type (str): 
                encoder type, can be `'ECA'` or `'ASP'`
            log_sinc (bool): 
                apply log in sinc layer
            norm_sinc (str): 
                apply norm in sinc layer, `norm_sinc` can be `'mean'` or `'mean_std'`
            sinc_stride (int): 
                stride size in sinc layer
            out_bn (bool): 
                apply batch normalization to output
            n_out (int): 
                output channels
        """
        super().__init__()

        assert norm_sinc in ['mean', 'mean_std'], f"Expect `norm_sinc` in ['mean', 'mean_std'], got {norm_sinc}"

        ### Set up arguments
        self.context = context
        self.encoder_type = encoder_type
        self.log_sinc = log_sinc
        self.norm_sinc = norm_sinc
        self.out_bn = out_bn
        self.summed = summed

        ### Preprocessing layers
        self.preprocess = nn.Sequential(
            PreEmphasis(),
            nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )

        ### All values of parameters are based on the paper above

        ### Convolutional encoder
        self.conv_1 = Encoder(
            ParamSincFB(
                n_filters=channels // 4,
                kernel_size=251,
                stride=sinc_stride
            )
        )
        self.bn_1 = nn.BatchNorm1d(channels // 4)
        self.relu = nn.ReLU()

        ### Res2MaxPooling blocks
        self.res2mp_1 = Res2MaxPoolingBlock(
            channels // 4, channels, kernel_size=3, dilation=2, scale=model_scale, max_pooling_size=5
        )

        self.res2mp_2 = Res2MaxPoolingBlock(
            channels,      channels, kernel_size=3, dilation=3, scale=model_scale, max_pooling_size=3
        )

        self.res2mp_3 = Res2MaxPoolingBlock(
            channels,      channels, kernel_size=3, dilation=4, scale=model_scale
        )

        res2mp_dim = 1536
        self.res2mp_end = nn.Conv1d(3 * channels, res2mp_dim, kernel_size=1)

        ### Attention (context and encoder type)
        attn_input_channels = res2mp_dim * 3 if self.context else res2mp_dim
        
        if self.encoder_type == "ECA":
            attn_output_channels = res2mp_dim
        elif self.encoder_type == "ASP":
            attn_output_channels = 1
        else:
            raise ValueError(f"Undefined encoder type. Expect 'ECA' or 'ASP', got {self.encoder_type}")
        
        attn_hidden_channels = 128
        self.attention = nn.Sequential(
            nn.Conv1d(attn_input_channels,  attn_hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attn_hidden_channels),   ### BN 4
            nn.Conv1d(attn_hidden_channels, attn_output_channels, kernel_size=1),
            nn.Softmax(dim=2)
        )

        ### Fully connected and max pooling
        self.bn_5 = nn.BatchNorm1d(res2mp_dim * 2)


        self.fc_6 = nn.Linear(res2mp_dim * 2, n_out)
        self.bn_6 = nn.BatchNorm1d(n_out)

        self.max_pooling = nn.MaxPool1d(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
            x (Tensor):
                input, shape is (batch_size, n_sample)
        """
        ### Proprocessing layers
        eps = 1e-6

        with torch.cuda.amp.autocast(enabled=False):
            x = self.preprocess(x)
            x = torch.abs(self.conv_1(x))

            if self.log_sinc:
                x = torch.log(x + eps)

            if self.norm_sinc == 'mean':
                x = x - torch.mean(x, dim=-1, keepdim=True)
            elif self.norm_sinc == 'mean_std':
                m = torch.mean(x, dim=-1, keepdim=True)
                s = torch.std( x, dim=-1, keepdim=True)
                s[s < 0.001] = 0.001
                x = (x - m) / s
            
        ### Res2MaxPooling blocks
        if self.summed:
            x1 = self.res2mp_1(x)
            x2 = self.res2mp_2(x1)
            x3 = self.res2mp_3(self.max_pooling(x1) + x2)
        else:
            x1 = self.res2mp_1(x)
            x2 = self.res2mp_2(x1)
            x3 = self.res2mp_3(x2)

        x = torch.cat([self.max_pooling(x1), x2, x3], dim=1)
        x = self.res2mp_end(x)
        x = self.relu(x)

        ### Attention
        t = x.size()[-1]
        if self.context:
            global_x = torch.cat([
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)).repeat(1, 1, t)
            ],
                dim=1
            )
        else:
            global_x = x
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sigma = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4, max=1e4))
        
        x = torch.cat([mu, sigma], dim=1)
        
        ### Fully connected layer and max pooling
        x = self.bn_5(x)
        x = self.fc_6(x)

        if self.out_bn:
            x = self.bn_6(x)

        return x


def model_init(**kwargs):
    return RawNet3(
        model_scale=8,
        context=True,
        summed=True,
        out_bn=False,
        norm_sinc='mean',
        log_sinc=True,
        grad_mult=1,
        **kwargs
    )