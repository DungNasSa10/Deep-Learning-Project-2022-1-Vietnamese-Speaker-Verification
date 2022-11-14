import os
import sys
sys.path.append(os.getcwd())

from src.crawling.vad_processor import VADProcessor

vad = VADProcessor()

print(vad.vad("../../data/wav/Chu-Văn-Biên/Nội dung chương trình Vật lí 11_Thầy Chu Văn Biên.wav", sampling_rate=16000))

