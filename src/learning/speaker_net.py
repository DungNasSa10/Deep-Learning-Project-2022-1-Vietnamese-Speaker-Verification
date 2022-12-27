import torch
import torch.nn as nn

import importlib


class WrappedModel(nn.Module):
    
    # The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model: nn.Module) -> None:
        
        super().__init__()
        self.module = model

    def forward(self, x, label = None):
        
        return self.module(x, label)


class SpeakerNet(nn.Module):

    def __init__(self, model: nn.Module, loss_function: str = "aamsoftmax", n_per_speaker: int = 1, device: str = 'cuda', **kwargs) -> None:
        
        super().__init__()

        speaker_model = importlib.import_module(f"learning.models.{model}").__getattribute__("model_init")
        self.__model__ = speaker_model(**kwargs)

        loss = importlib.import_module(f"learning.losses.{loss_function}").__getattribute__("loss_init")
        self.__loss__ = loss(**kwargs)

        self.n_per_speaker = n_per_speaker

        self.device = device

    def forward(self, data: torch.tensor, label = None):
        print("device", self.device)
        data = data.reshape(-1, data.size()[-1]).to(self.device)
        output = self.__model__(data)

        if label is None:
            return output
        else:
            output = output.reshape(self.n_per_speaker, -1, output.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec1 = self.__loss__(output, label)

            return nloss, prec1
