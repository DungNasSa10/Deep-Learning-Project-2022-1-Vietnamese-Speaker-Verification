import os
from utils.prepare_data import get_voices_and_urls
from crawling.mixin import StepMixin
from crawling.downloader import Downloader
from crawling.vad_processor import VADProcessor
from crawling.outlier_remover import OutlierRemover


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
        self.logger.info("Start full process")

        voice_and_urls = get_voices_and_urls(csv_voice_filepath)
        fn, _ = os.path.splitext(csv_voice_filepath)
        fn = fn + "-urls-check-list.txt"
        if os.path.exists(fn) is False:
            self.logger.info(f"Create urls check list at file: {fn}")
            f = open(fn, 'w', encoding='utf-8')
            f.close()
        
        def get_urls():
            f = open(fn, 'r', encoding='utf-8')
            urls = f.read().split()
            f.close()
            return urls

        def insert_url(url):
            f = open(fn, 'a', encoding='utf-8')
            f.write(url + '\n')
            f.close()

        for v, urls in voice_and_urls:
            self.logger.info("Start downloading videos of voice: " + v)
            path = os.path.join(save_dir, v)
            if os.path.exists(path) is False:
                os.makedirs(path)
                self.logger.warning("Create directory " + path)
            
            for url in urls:
                if url in get_urls():
                    self.logger.warning(f"Skip url {url} because it has finished the process")
                    continue

                wav_path = self.downloader.run(url, save_dir=path, sampling_rate=wav_sample_rate, remove_mp3=remove_mp3)
                wav_dir = self.vad_processor.run(wav_path, sampling_rate=wav_sample_rate)
                self.outlier_remover.run(wav_dir)

                insert_url(url)
            self.logger.info("Finish processing for voice: " + v)

        self.logger.info("Finish full process")