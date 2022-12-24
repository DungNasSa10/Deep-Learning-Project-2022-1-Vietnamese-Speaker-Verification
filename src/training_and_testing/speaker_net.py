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

    def __init__(self, model: nn.Module, loss_function: str = "aamsoftmax", n_per_speaker: int = 1, **kwargs) -> None:
        
        super().__init__()

        speaker_model = importlib.import_module(f"models.{model}").__getattribute__("model")
        self.__model__ = speaker_model(**kwargs)

        loss = importlib.import_module(f"losses.{loss_function}").__getattribute__("loss_function")
        self.__loss__ = loss(**kwargs)

    def forward(self, data: torch.tensor, label = None):
        
        data = data.reshape(-1, data.size()[-1]).cuda()
        output = self.__model(data)

        if label == None:
            return output
        else:
            output = output.reshape(self.n_per_speaker, -1, output.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec1 = self.__L__.forward(output, label)

            return nloss, prec1