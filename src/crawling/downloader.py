import os
import subprocess
import librosa
import soundfile
from pytube import YouTube
from typing import Tuple, List
from ..utils.logger import get_logger
from ..utils.prepare_data import get_voices_and_urls


class Downloader:
    """
    Downloader save directory tree:
        data
            wavs
                <voice name 1>
                    *.wav
                    *.mp3
                <voice name 2>
                    *.wav
                    *.mp3
    """

    def __init__(self) -> None:
        self.logger = get_logger("Downloader")

    def download_mp3(self, url: str, save_dir: str) -> Tuple[str, bool]:
        """
        Args:
            url: url of Youtube video
            save_dir: save directory
        
        Returns:
            file path of mp3 file and download status, True is downloaded, else False \
            if return None => there is no mp3 downloaded
        """
        yt = YouTube(url)
        
        ### extract only audio
        audio = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        audio_name = audio.default_filename.split('.')[0]

        if os.path.exists(os.path.join(save_dir, audio_name + '.wav')):
            self.logger.warning("Skip downloading audio at " + url + " because the audio has been downloaded")
            return os.path.join(save_dir, audio_name + '.wav'), False

        if os.path.exists(os.path.join(save_dir, audio_name + '.mp3')):
            self.logger.warning("Skip downloading audio at " + url + " because the audio has been downloaded")
            return os.path.join(save_dir, audio_name + '.mp3'), False

        else:
            ### download the file
            self.logger.info("Start downloading audio at " + url + " ...")
            out_path = audio.download(output_path=save_dir)
            
            ### save the file
            base, _ = os.path.splitext(out_path)
            mp3_path = base + '.mp3'
            os.rename(out_path, mp3_path)

            self.logger.info('Downloaded audio successfully, store at ' + mp3_path)

        return mp3_path, True

    def download_multiple_mp3(self, urls: List[str], save_dir: str) -> List[str]:
        """
        Args:
            urls: list of Youtube video url
            save_dir: save directory

        Returns:
            list of mp3 file paths
        """
        return [self.download_mp3(url, save_dir) for url in urls]

    @staticmethod
    def mp3_to_wav(mp3_path: str, wav_path: str=None) -> str:
        """
        Args:
            mp3_path: path to your mp3 file
            wav_path: path to your converted destination wav file \
                if None then the wav file will be named the same as mp3 file and be stored in the same directory

        Returns:
            converted destination wav file
        """
        wav_path = os.path.splitext(mp3_path)[0] + '.wav' if wav_path is None else wav_path
        subprocess.call(['ffmpeg', '-i', mp3_path, wav_path])
        return wav_path

    @staticmethod
    def resample_wav(wav_path: str, save_path: str=None, target_sr=16000):
        """
        Args:
            wav_path: path to your wave file
            save_path: path to save you resampled wav
            target_sr: target sample rate
        """
        save_path   = wav_path if save_path is None else save_path
        y, sr       = librosa.load(wav_path)       
        y_k         = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        y_mono      = librosa.to_mono(y_k)
        soundfile.write(save_path, y_mono, target_sr)

    def download(self, urls: List[str], save_dir: str, download_first: bool, remove_mp3: bool, wav_sample_rate=16000):
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
            self.logger.warning("Create directory " + save_dir)

        def exec(mp3_path):
            if os.path.splitext(mp3_path)[-1] == '.mp3':
                self.logger.info("Convert and resample audio")
                self.resample_wav(self.mp3_to_wav(mp3_path), target_sr=wav_sample_rate)
                os.remove(mp3_path)
                if remove_mp3:
                    self.logger.info("Remove " + mp3_path)

        if download_first:
            downloaded_mp3_paths = self.download_multiple_mp3(urls, save_dir)
            for mp3_path, status in downloaded_mp3_paths:
                exec(mp3_path)

        else:
            for url in urls:
                mp3_path, status = self.download_mp3(url, save_dir)
                exec(mp3_path)

    def run(self, csv_voice_filepath: str, save_dir: str, download_first: bool, remove_mp3: bool, wav_sample_rate=16000):
        voice_and_urls = get_voices_and_urls(csv_voice_filepath)

        for v, urls in voice_and_urls:
            self.logger.info("Start downloading videos of voice: " + v + " ...")
            path = os.path.join(save_dir, v)
            if os.path.exists(path) is False:
                os.makedirs(path)
                self.logger.warning("Create directory " + path)
            
            self.download(urls, save_dir=path, download_first=download_first, remove_mp3=remove_mp3, wav_sample_rate=wav_sample_rate)