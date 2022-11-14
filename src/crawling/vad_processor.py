from torch import hub
from glob import glob
import os
import soundfile


class VADProcessor:
    def __init__(self) -> None:
        self.model, utils = hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True)
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils

    def vad(self, wav_path: str, sampling_rate: int):
        j = 0
        wav = self.read_audio(wav_path, sampling_rate = sampling_rate)
        speech_timestamps = self.get_speech_timestamps(wav, self.model, threshold=0.5, sampling_rate=sampling_rate)

        wav_duration = 0
        k = 0   
        speech_timestamps_concat = []
        wav_concat = []

        while(True):
            wav_duration += speech_timestamps[k]['end'] - speech_timestamps[k]['start'] 
            speech_timestamps_concat.append(speech_timestamps[k])
                
            if k < len(speech_timestamps)-1:
                k += 1
                if wav_duration >= 3*16000:      
                    wav_concat.append(self.collect_chunks(speech_timestamps_concat, wav))
                    speech_timestamps_concat.clear()
                    wav_duration = 0
                    continue
                else:
                    continue
            else:
                wav_concat.append(self.collect_chunks(speech_timestamps_concat, wav))
                speech_timestamps_concat.clear()
                wav_duration = 0
                break   

        wav_dir = os.path.splitext(wav_path)[0] + '/'
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

        for i in wav_concat:
            if not os.path.exists(wav_dir + str(j) + '.wav'):
                self.save_audio(wav_dir + str(j) + '.wav', i, sampling_rate = sampling_rate)
            j += 1 
        
        return wav_dir

    def re_vad(self, wavs_dir: str, sampling_rate: int):
        files = glob(wavs_dir + '*.wav')
        j = len(files) + 1

        for file in files:
            y, sr = soundfile.read(file)
            wav_length = len(y) / sr

            if wav_length >= 10:
                wav = self.read_audio(file, sampling_rate = sampling_rate)
                speech_timestamps = self.get_speech_timestamps(wav, self.model, threshold=0.9, sampling_rate=sampling_rate)
                wav_duration = 0
                k = 0   
                speech_timestamps_concat = []
                wav_concat = []

                if len(speech_timestamps) != 0:
                    while(True):
                        wav_duration =  wav_duration + speech_timestamps[k]['end'] - speech_timestamps[k]['start']
                        
                        speech_timestamps_concat.append(speech_timestamps[k])
                        
                        if k < len(speech_timestamps)-1:
                            k += 1
                            if wav_duration >= 3*16000:      
                                wav_concat.append(self.collect_chunks(speech_timestamps_concat, wav))
                                speech_timestamps_concat.clear()
                                wav_duration = 0
                                continue
                            else:
                                continue
                        else:
                            wav_concat.append(self.collect_chunks(speech_timestamps_concat, wav))
                            speech_timestamps_concat.clear()
                            wav_duration = 0
                            break
                else:
                    continue   
                            
                for i in wav_concat:
                    self.save_audio(wavs_dir + str(j) + '.wav', i, sampling_rate = sampling_rate)
                    j += 1 