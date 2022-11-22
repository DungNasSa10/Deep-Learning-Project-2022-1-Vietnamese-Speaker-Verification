from torch import hub
from utils.prepare_data import get_wav_files
import os
import soundfile
from crawling.mixin import StepMixin


class VADProcessor(StepMixin):
    def __init__(self) -> None:
        super().__init__() 
        
        self.model, utils = hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils

        self.min_duration_in_seconds = 3
        self.max_duration_in_seconds = 10
        self.thresholds = (0.5, 0.9)

    def vad(self, wav, sampling_rate: int, threshold: int, min_duration: int):
        """
        Args:
            min_duration: int
                min duration of splitted wav file, in seconds
        """
        min_duration *= sampling_rate 
        speech_timestamps_chunks = []
        wav_chunks = []
        duration = 0
        speech_timestamps = self.get_speech_timestamps(wav, self.model, threshold=threshold, sampling_rate=sampling_rate)

        for ts in speech_timestamps:
            duration += ts['end'] - ts['start']
            speech_timestamps_chunks.append(ts)
            if duration >= min_duration:
                wav_chunks.append(speech_timestamps_chunks)
                speech_timestamps_chunks = []
                duration = 0

        if duration > 0:
            if len(wav_chunks) > 0:
                wav_chunks.append(wav_chunks.pop() + speech_timestamps_chunks)
            else:
                wav_chunks.append(speech_timestamps_chunks)
    
        return [self.collect_chunks(chunks, wav) for chunks in wav_chunks]

    def remove_wav(self, wav_dir: str) -> str:
        self.logger.info("Start removing some unnecessary .wav file in directory: " + wav_dir)
        files = get_wav_files(wav_dir)
        self.logger.info(f"Detect {len(files)} .wav files in {wav_dir}")

        for file in files:
            y, sr = soundfile.read(file)
            wav_length = len(y) / sr

            if wav_length < self.min_duration_in_seconds or wav_length > self.max_duration_in_seconds:
                self.logger.warning("Remove " + file)
                os.remove(file)
        
        self.logger.info("Finish removing some unnecessary .wav file")
        return wav_dir

    def run(self, wav_path: str, sampling_rate: int=16000) -> str:
        """
        Args:
        
        Returns:
            directory where wav files are stored
        """
        self.logger.info("Start running VAD process with file: " + wav_path)
        
        wav = self.read_audio(wav_path, sampling_rate = sampling_rate)
        wav_dir = os.path.splitext(wav_path)[0]
        if os.path.exists(wav_dir) is False:
            os.makedirs(wav_dir)

        wav_chunks = self.vad(wav, sampling_rate, self.thresholds[0], self.min_duration_in_seconds)
        wav_index = 0
        max_duration = self.max_duration_in_seconds * sampling_rate

        def save_chunk(_chunk, index):
            fpath = os.path.join(wav_dir, str(index) + ".wav")
            self.logger.info('Save audio ' + fpath)
            self.save_audio(fpath, _chunk, sampling_rate=sampling_rate)
            return index + 1

        for chunk in wav_chunks:
            if len(chunk) < max_duration:
                wav_index = save_chunk(chunk, wav_index)
            else:
                chunks = self.vad(chunk, sampling_rate, self.thresholds[1], self.min_duration_in_seconds)
                for ch in chunks:
                    wav_index = save_chunk(ch, wav_index)

        self.remove_wav(wav_dir)
        self.logger.warning("Remove file: " + wav_path)
        os.remove(wav_path)

        self.logger.info("Finish VAD process")

        return wav_dir