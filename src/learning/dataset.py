import torch
import numpy as np
import random
import os
import glob
import tqdm
import soundfile, librosa
from scipy import signal
from torch.utils.data import Dataset
import torch.distributed as dist


def round_down(num, divisor):
	return num - (num%divisor)

def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

	# Maximum audio length
	max_audio = max_frames * 160 + 240

	# Read wav file and convert to torch tensor
	audio, sample_rate = soundfile.read(filename)

	audiosize = audio.shape[0]

	if audiosize <= max_audio:
		shortage    = max_audio - audiosize + 1 
		audio       = np.pad(audio, (0, shortage), 'wrap')
		audiosize   = audio.shape[0]

	if evalmode:
		startframe = np.linspace(0,audiosize-max_audio, num=num_eval)
	else:
		startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
	
	feats = []
	if evalmode and max_frames == 0:
		feats.append(audio)
	else:
		for asf in startframe:
			feats.append(audio[int(asf):int(asf)+max_audio])

	feat = np.stack(feats,axis=0).astype(np.float)

	return feat
	

class AugmentWAV(object):

	def __init__(self, musan_path, rir_path, max_frames):

		self.max_frames = max_frames
		self.max_audio  =  max_frames * 160 + 240

		self.noisetypes = ['noise', 'music']

		self.noisesnr   = {'noise':[0,15], 'music':[5,15]}
		self.numnoise   = {'noise':[1,1], 'music':[1,1] }
		self.noiselist  = {}

		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))

		for file in augment_files:
			if not file.split('/')[-3] in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			self.noiselist[file.split('/')[-3]].append(file)

		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

		self.encoded_audio=dict()
		self.encoded_audio["noise"]=[]
		self.encoded_audio["music"]=[]
				
		for noisecat in self.noisetypes:
			for noise in tqdm.tqdm(self.noiselist[noisecat]):
				noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
				noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
				noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
				self.encoded_audio[noisecat].append((noise_db, noise_snr, noiseaudio))

	def additive_noise(self, noisecat, audio):

		clean_db    = 10 * np.log10(np.mean(audio ** 2)+1e-4)

		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.encoded_audio[noisecat], random.randint(numnoise[0], numnoise[1]))
		noises = []
		for noise in noiselist:
			noises.append(np.sqrt(10 ** ((clean_db - noise[0] - noise[1]) / 10)) * noise[2])

		noise = np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio

	def reverberate(self, audio):

		rir_file    = random.choice(self.rir_files)
		rir, sr     = librosa.load(rir_file, sr=None, mono=True)
		rir         = np.expand_dims(rir.astype(np.float),0)
		rir         = rir / np.sqrt(np.sum(rir**2))

		return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

	def forward(self, audio):
		if isinstance(audio, torch.Tensor):
			np_audio = audio.cpu().np()
		augtype = random.randint(1, 3)

		if augtype   == 1:
			new_audio   = self.reverberate(np_audio)
		elif augtype == 2: 
			new_audio   = self.additive_noise('music', np_audio)
		elif augtype == 3:
			new_audio   = self.additive_noise('noise', np_audio)

		return torch.FloatTensor(new_audio)


class TrainDataset(Dataset):
	def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path, **kwargs):

		self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

		self.train_list = train_list
		self.max_frames = max_frames
		self.musan_path = musan_path
		self.rir_path   = rir_path
		self.augment    = augment
		
		# Read training files
		with open(train_list) as dataset_file:
			lines = dataset_file.readlines()

		# Make a dictionary of ID names and ID indices
		dictkeys = list(set([x.split("\t")[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

		# Parse the training list into file names and ID indices
		self.data_list  = []
		self.data_label = []
		
		for lidx, line in enumerate(lines):
			data = line.strip().split("\t")

			speaker_label = dictkeys[data[0]]
			filename = os.path.join(train_path,data[1])
			
			self.data_label.append(speaker_label)
			self.data_list.append(filename)

	def __getitem__(self, indices):

		feat = []

		for index in indices:
			
			audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
			
			if self.augment:
				aug_prob = random.uniform(0, 1)

				if "(" in self.data_list[index]:
					aug = aug_prob < 0.3
				else:
					aug = aug_prob < 0.65
				
				if aug:
					augtype = random.randint(0,3)
					if augtype == 1:
						audio   = self.augment_wav.reverberate(audio)
					elif augtype == 2:
						audio   = self.augment_wav.additive_noise('music',audio)
					elif augtype == 3:
						audio   = self.augment_wav.additive_noise('noise',audio)
					
			feat.append(audio)

		feat = np.concatenate(feat, axis=0)

		return torch.FloatTensor(feat), self.data_label[index]

	def __len__(self):
		return len(self.data_list)


class TestDataset(Dataset):
	def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
		self.max_frames = eval_frames
		self.num_eval   = num_eval
		self.test_path  = test_path
		self.test_list  = test_list

	def __getitem__(self, index):
		audio = loadWAV(os.path.join(self.test_path,self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
		return torch.FloatTensor(audio), self.test_list[index]

	def __len__(self):
		return len(self.test_list)


class TrainDataSampler(torch.utils.data.Sampler):
	def __init__(self, data_source, batch_size, seed, n_per_speaker=1, max_seg_per_spk=500, distributed=False, **kwargs):

		self.data_label         = data_source.data_label
		self.n_per_speaker        = n_per_speaker
		self.max_seg_per_spk    = max_seg_per_spk
		self.batch_size         = batch_size
		self.epoch              = 0
		self.seed               = seed
		self.distributed        = distributed
		
	def __iter__(self):

		g = torch.Generator()
		g.manual_seed(self.seed + self.epoch)
		indices = torch.randperm(len(self.data_label), generator=g).tolist()

		data_dict = {}

		# Sort into dictionary of file indices for each ID
		for index in indices:
			speaker_label = self.data_label[index]
			if not (speaker_label in data_dict):
				data_dict[speaker_label] = []
			data_dict[speaker_label].append(index)

		## Group file indices for each class
		dictkeys = list(data_dict.keys())
		dictkeys.sort()

		lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

		flattened_list = []
		flattened_label = []
		
		for findex, key in enumerate(dictkeys):
			data    = data_dict[key]
			numSeg  = round_down(min(len(data), self.max_seg_per_spk), self.n_per_speaker)
			
			rp      = lol(np.arange(numSeg), self.n_per_speaker)
			flattened_label.extend([findex] * (len(rp)))
			for indices in rp:
				flattened_list.append([data[i] for i in indices])

		## Mix data in random order
		mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
		mixlabel        = []
		mixmap          = []

		## Prevent two pairs of the same speaker in the same batch
		for ii in mixid:
			startbatch = round_down(len(mixlabel), self.batch_size)
			if flattened_label[ii] not in mixlabel[startbatch:]:
				mixlabel.append(flattened_label[ii])
				mixmap.append(ii)

		mixed_list = [flattened_list[i] for i in mixmap]

		## Divide data to each GPU
		if self.distributed:
			total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
			start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
			end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
			self.num_samples = end_index - start_index
			return iter(mixed_list[start_index:end_index])
		else:
			total_size = round_down(len(mixed_list), self.batch_size)
			self.num_samples = total_size
			return iter(mixed_list[:total_size])

	
	def __len__(self) -> int:
		return self.num_samples

	def set_epoch(self, epoch: int) -> None:
		self.epoch = epoch