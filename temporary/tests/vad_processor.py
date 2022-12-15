import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.crawling.vad_processor import VADProcessor
from src.crawling.downloader import Downloader

d = Downloader()
vad = VADProcessor()

wav_path = d.run('https://www.youtube.com/watch?v=dwrQeJLsl5Y&list=PLmyF-BPWWPTIXy_kCK5GvCZeZye_5Ny86&index=3', './data/wavs')
print(vad.run(wav_path, sampling_rate=16000))

