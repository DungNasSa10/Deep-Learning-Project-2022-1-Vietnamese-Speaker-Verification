from speechbrain.pretrained import EncoderClassifier
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np
from copy import deepcopy
import torchaudio
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class OutlierRemover:
    pass