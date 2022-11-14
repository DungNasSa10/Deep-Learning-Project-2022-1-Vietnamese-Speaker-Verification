import os
import sys
sys.path.append(os.getcwd())

from src.crawling.vad_processor import VADProcessor

vad = VADProcessor()

print(vad.vad("./data/wavs/ED/Ed Sheeran - Bad Habits [Official Video].wav", sampling_rate=16000))

