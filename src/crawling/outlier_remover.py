import os
from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.pretrained import EncoderClassifier

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .mixin import StepMixin
from ..utils.prepare_data import get_wav_files


class OutlierRemover(StepMixin):
    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Detect device: " + str(self.device))
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":self.device})

    def run(self, wav_dir: str, threshold: float=0.45):
        """
        Args:
            wav_dir:
                directory of .wav files
                the .wav files inside this directory will be 
            threshold: float \
                if score of .wav file in `wav_dir` directory is less than this threshold then it will be considered as the outlier
                and will be removed
        """
        self.logger.info("Start outlier-removing process")
        # files = glob(os.path.join(wav_dir, '*.wav'))
        files = get_wav_files(wav_dir)
        embeddings = {}
        self.logger.info(f"Detect {len(files)} .wav files in {wav_dir}")

        self.logger.info("Calculate the embedding values of audio")
        with logging_redirect_tqdm():
            for idx, file in tqdm(enumerate(files), total=len(files)):
                signal, _  = torchaudio.load(file)

                # Splited utterance matrix
                max_audio = 3*16000
                feats = []
                startframe = np.linspace(0, signal.shape[1] - max_audio, num=5)

                for asf in startframe:
                    feat = signal[0, int(asf): int(asf) + max_audio]
                    feats.append(feat)

                feats = np.stack(feats, axis = 0).astype('f')
                data = torch.FloatTensor(feats)

                # Speaker embeddings
                with torch.no_grad():
                    embedding_1 = self.model.encode_batch(signal.to(self.device))
                    embedding_1 = F.normalize(embedding_1[0, 0, :], p=2, dim=0)
                    embedding_2 = self.model.encode_batch(data.to(self.device))
                    embedding_2 = F.normalize(embedding_2.permute(1, 0, 2)[0, :, :], p=2, dim=1)

                    embeddings[file] = [embedding_1, embedding_2]

        self.logger.info("Score the embedding")
        scores = np.array([ [0]*(len(files) - 1) for i in range(len(files)) ], dtype='f')

        with logging_redirect_tqdm():
            for i, file in tqdm(enumerate(files), total=len(files)):
                remaining_files = deepcopy(files)
                remaining_files.remove(file)
                embedding_11, embedding_12 = embeddings[file]

                for j in range(len(remaining_files)):
                    embedding_21, embedding_22 = embeddings[remaining_files[j]]
                    # Compute the scores
                    score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
                    score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
                    score = (score_1 + score_2) / 2
                    score = score.detach().cpu().numpy()
                    scores[i, j] = score

        mean_scores = np.mean(scores, axis=1)
        
        self.logger.info("Remove outliers")
        with logging_redirect_tqdm():
            for i, file in tqdm(enumerate(files), total=len(files)):
                if mean_scores[i] < threshold:
                    # print(mean_scores[i], file)
                    self.logger.warning("Remove " + file)
                    os.remove(file)