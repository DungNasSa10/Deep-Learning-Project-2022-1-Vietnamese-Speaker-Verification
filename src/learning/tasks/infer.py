import tqdm
import soundfile
import torch
import torch.nn.functional as F
from learning.speaker_net import SpeakerNet, WrappedModel
from learning.tasks.model_controller import ModelController
import numpy as np



def infer(model: str = "SEResNet34", 
            pretrained_checkpoint: str = "../output/model_checkpoints/SERetNet34_AAM.model", 
            wav_path_1: str = "../data/test/sv_vlsp_2021/public_test/competition_public_test/aff28b55-f45e-459a-a317-98e874b8cd62.wav", 
            wav_path_2: str = "../data/test/sv_vlsp_2021/public_test/competition_public_test/ce609317-2731-4ec5-a2b0-f6ec7a25ab81.wav", 
        ) -> float:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Detect device:", device)

    # Create model
    if model == "VGGVox":
        speaker_model = SpeakerNet(model=model, device=device, n_mels=40, log_inputs=False, encoder_type="SAP", n_out=512)
    else:
        speaker_model = SpeakerNet(model=model, device=device, n_mels=80, log_inputs=False, encoder_type="ASP", sinc_stride=10, C=1024, n_out=512)
        
    speaker_model = WrappedModel(speaker_model).to(device)

    # Create controller
    controller = ModelController(speaker_model=speaker_model, device=device)

    # Load pretrained model
    controller.loadParameters(pretrained_checkpoint)
    
    # Inference
    controller.__model__.eval()
    files = [wav_path_1, wav_path_2]
    embeddings = {}
    setfiles = list(set(files))

    for i, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
        audio, _  = soundfile.read(file)
        # Full utterance
        data_1 = torch.FloatTensor(np.stack([audio],axis=0)).to(controller.device)

        # Splited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf): int(asf)+max_audio])
        feats = np.stack(feats, axis = 0).astype(float)
        data_2 = torch.FloatTensor(feats).to(controller.device)
        # Speaker embeddings
        with torch.no_grad():
            embedding_1 = controller.__model__(data_1)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = controller.__model__(data_2)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        embeddings[file] = [embedding_1, embedding_2]
		
    embedding_11, embedding_12 = embeddings[wav_path_1]
    embedding_21, embedding_22 = embeddings[wav_path_2]
    
    # Compute the scores
    score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) 
    score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
    score = (score_1 + score_2) / 2
    score = score.detach().cpu().numpy()

    return score