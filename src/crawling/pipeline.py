import os
from ..utils.prepare_data import get_voices_and_urls
from .mixin import StepMixin
from .downloader import Downloader
from .vad_processor import VADProcessor
from .outlier_remover import OutlierRemover


class Pipeline(StepMixin):
    def __init__(self) -> None:
        super().__init__() 

        self.downloader = Downloader()
        self.vad_processor = VADProcessor()
        self.outlier_remover = OutlierRemover()

    def run(self, csv_voice_filepath: str, save_dir: str='./data/wavs', remove_mp3: bool=True, wav_sample_rate: int=16000):
        """
        Args:
            csv_voice_filepath: str
            save_dir: str \
                directory to save .wav file
            remove_mp3: 
                remove old mp3 files or not
            wav_sample_rate: int, default = 16000 \
                sample rate of .wav file
        """
        voice_and_urls = get_voices_and_urls(csv_voice_filepath)

        for v, urls in voice_and_urls:
            self.logger.info("Start downloading videos of voice: " + v)
            path = os.path.join(save_dir, v)
            if os.path.exists(path) is False:
                os.makedirs(path)
                self.logger.warning("Create directory " + path)
            
            for url in urls:
                wav_path = self.downloader.run(url, save_dir=path, sampling_rate=wav_sample_rate, remove_mp3=remove_mp3)
                wav_dir = self.vad_processor.run(wav_path, sampling_rate=wav_sample_rate)
                self.outlier_remover.run(wav_dir)
                return