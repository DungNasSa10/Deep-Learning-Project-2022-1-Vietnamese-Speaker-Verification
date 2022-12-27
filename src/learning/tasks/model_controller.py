import numpy as np
import importlib, sys, time, tqdm, soundfile, os

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from learning.speaker_net import WrappedModel


class ModelController(object):
    def __init__(self, speaker_model: WrappedModel, optimizer = "adam", scheduler = "steplr", device = "cuda", gpu = 0, mixedprec = False, **kwargs):

        self.__model__ = speaker_model

        optimizer = importlib.import_module(f"learning.optimizers.{optimizer}").__getattribute__("optimizer_init")
        self.__optimizer__ = optimizer(self.__model__.parameters(), **kwargs)

        scheduler = importlib.import_module(f"learning.schedulers.{scheduler}").__getattribute__("scheduler_init")
        self.__scheduler__, self.lr_step = scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.device = device
        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0

        # EER or accuracy
        for data, data_label in loader:

            data = data.transpose(1, 0)

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).to(self.device)

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            counter += 1
            index += stepsize

            if verbose:
                sys.stderr.write("Training {:d} / {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                                " Loss %.5f, TAcc %5f, LR %.7f \r" %(loss/counter, top1/counter, max([x['lr'] for x in self.__optimizer__.param_groups])))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def eval_network(self, test_list, test_path, **kwargs):
        self.__model__.eval()
        files = []
        embeddings = {}
        lines = open(test_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _  = soundfile.read(os.path.join(test_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(np.stack([audio],axis=0)).to(self.device)

            # Splited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = np.stack(feats, axis = 0).astype(np.float)
            data_2 = torch.FloatTensor(feats).to(self.device)
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.__model__(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.__model__(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels  = [], []

        for line in lines:			
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        return scores, labels

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def test_from_list(self, test_list, test_path, output_path, **kwargs):
        self.__model__.eval()
        files = []
        filename = test_list.split("/")[-1]
        os.makedirs(output_path, exist_ok=True)
        f_write = open(os.path.join(output_path, filename), "w")
        embeddings = {}
        lines = open(test_list).read().splitlines()
        
        for line in lines:
            files.append(line.split()[0])
            files.append(line.split()[1])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _  = soundfile.read(os.path.join(test_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(np.stack([audio],axis=0)).to(self.device)

            # Splited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = np.stack(feats, axis = 0).astype(float)
            data_2 = torch.FloatTensor(feats).to(self.device)
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.__model__(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.__model__(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        for line in lines:			
            embedding_11, embedding_12 = embeddings[line.split()[0]]
            embedding_21, embedding_22 = embeddings[line.split()[1]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) 
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            
            f_write.write(line.split()[0] + '\t' + line.split()[1] + '\t' + str(score) + '\n')

        f_write.close()

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location=f"cuda:{self.gpu}" if self.device == "cuda" else torch.device('cpu'))
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)