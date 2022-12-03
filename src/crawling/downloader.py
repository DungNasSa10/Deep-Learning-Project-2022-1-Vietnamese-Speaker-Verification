import os
import subprocess
import librosa
import soundfile
from pydub import AudioSegment
from pytube import YouTube
from typing import Tuple
from crawling.mixin import StepMixin


class Downloader(StepMixin):
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

    MAX_TIME_IN_MIN = 25

    def __init__(self) -> None:
        super().__init__()

    def trim_mp3(self, mp3_path: str, max_time_in_min: float) -> None:
        """ Trim mp3 file if it is very long

        Args:
            out_path: path of mp3 file
            max_time_in_minutes: max time in minutes of mp3 file
        """
        end_time = max_time_in_min * 60
        base, _ = os.path.splitext(mp3_path)
        trimmed_mp3_path = base + '_trimmed.mp3'
        subprocess.call(['ffmpeg', '-ss', '0', '-t', f'{end_time}', '-i', mp3_path, trimmed_mp3_path])
        os.remove(mp3_path)

        return trimmed_mp3_path

    def download_mp3(self, url: str, save_dir: str, max_time_in_min: float) -> Tuple[str, bool]:
        """
        Args:
            url: url of Youtube video
            save_dir: save directory
            max_time_in_min: max time of mp3 file in minutes

        Returns:
            file path of mp3 file
        """
        yt = YouTube(url)
        
        ### extract only audio
        audio = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        audio_name = audio.default_filename.split('.')[0]

        if os.path.exists(os.path.join(save_dir, audio_name + '.wav')):
            fp = os.path.join(save_dir, audio_name + '.wav')
            self.logger.warning("Remove old file: " + fp)
            os.remove(fp)

        if os.path.exists(os.path.join(save_dir, audio_name + '.mp3')):
            fp = os.path.join(save_dir, audio_name + '.mp3')
            self.logger.warning("Remove old file: " + fp)
            os.remove(fp)

        ### download the file
        self.logger.info("Start downloading audio at " + url)
        out_path = audio.download(output_path=save_dir)
        
        ### save the file
        base, _ = os.path.splitext(out_path)
        mp3_path = base + '.mp3'
        self.logger.warning(f"Rename {out_path} to {mp3_path}")
        os.rename(out_path, mp3_path)

        self.logger.info('Downloaded audio successfully, store at ' + mp3_path)

        ### trim mp3 file
        if yt.length > Downloader.MAX_TIME_IN_MIN * 60:
            trimmed_mp3_path = self.trim_mp3(mp3_path, max_time_in_min)
            
            return trimmed_mp3_path

        return mp3_path

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
            save_path: path to save you resampled wav \
                If None then overwrite file to wav_path
            target_sr: target sample rate

        Returns:
            path to resampled wav file
        """
        save_path   = wav_path if save_path is None else save_path
        y, sr       = librosa.load(wav_path)       
        y_k         = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        y_mono      = librosa.to_mono(y_k)
        soundfile.write(save_path, y_mono, target_sr)
        return save_path

    def run(self, url: str, save_dir: str, sampling_rate: int=16000, remove_mp3: bool=True):
        """
        Returns:
            path to downloaded .wav file
        """
        mp3_path = self.download_mp3(url, save_dir, Downloader.MAX_TIME_IN_MIN)
        if os.path.splitext(mp3_path)[-1] == '.mp3':
            self.logger.info("Convert into .wav and resample audio")
            save_path = self.resample_wav(self.mp3_to_wav(mp3_path), target_sr=sampling_rate)

            if remove_mp3:
                self.logger.info("Remove " + mp3_path)
                os.remove(mp3_path)
        
            return save_path
        return mp3_path