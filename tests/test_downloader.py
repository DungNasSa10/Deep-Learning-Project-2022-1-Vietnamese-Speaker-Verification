import os
import sys

sys.path.append(os.getcwd())

from src.crawling.downloader import Downloader


d = Downloader()

d.run(['https://www.youtube.com/watch?v=orJSJGHjBLI'], save_dir="data/wavs", download_first=False, remove_mp3=True)
