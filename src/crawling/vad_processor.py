from torch import hub
from ..utils.prepare_data import get_wav_files
import os
import soundfile
from .mixin import StepMixin


class VADProcessor(StepMixin):
    def __init__(self) -> None:
        super().__init__() 
        
        self.model, utils = hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils

    def vad(self, wav_path: str, sampling_rate: int=16000):
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
                if wav_duration >= 3 * sampling_rate:      
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

        wav_dir = os.path.splitext(wav_path)[0]
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)
            self.logger.warning("Create directory: " + wav_dir)

        j = 0
        self.logger.info("Start saving .wav files to " + wav_dir)
        for i in wav_concat:
            if not os.path.exists(os.path.join(wav_dir, str(j) + '.wav')):
                self.logger.info("Save file " + os.path.join(wav_dir, str(j) + '.wav'))
                self.save_audio(os.path.join(wav_dir, str(j) + '.wav'), i, sampling_rate = sampling_rate)
            j += 1 

        self.logger.info("Save .wav files successfully")
        
        files = get_wav_files(wav_dir)
        j = len(files) + 1
        self.logger.info(f"Detect {len(files)} .wav files in {wav_dir}")

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
                            if wav_duration >= 3 * sampling_rate:      
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

                self.logger.info("Start saving .wav files to " + wav_dir)
                for i in wav_concat:
                    self.logger.info("Save file " + str(j) + '.wav')
                    self.save_audio(os.path.join(wav_dir, str(j) + '.wav'), i, sampling_rate = sampling_rate)
                    j += 1 

        return wav_dir

    def remove_wav(self, wav_dir: str) -> str:
        self.logger.info("Remove some unnecessary .wav file")
        files = get_wav_files(wav_dir)
        self.logger.info(f"Detect {len(files)} .wav files in {wav_dir}")

        for file in files:
            y, sr = soundfile.read(file)
            wav_length = len(y) / sr

            if wav_length < 3 or wav_length > 10:
                os.remove(file)
                self.logger.warning("Remove " + file)

        return wav_dir

    def run(self, wav_path: str, sampling_rate: int=16000) -> str:
        """
        Args:
        
        Returns:
            directory where wav files are stored
        """
        self.logger.info("Start running VAD process with file: " + wav_path)
        wav_dir = self.remove_wav(self.vad(wav_path, sampling_rate=sampling_rate))
        self.logger.info("Finish VAD process")

        return wav_dir