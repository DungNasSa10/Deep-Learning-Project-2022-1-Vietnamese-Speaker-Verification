import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.crawling.downloader import Downloader


d = Downloader()

# d.download(['https://www.youtube.com/watch?v=orJSJGHjBLI'], save_dir="data/wavs", download_first=False, remove_mp3=True)
d.run("https://www.youtube.com/watch?v=orJSJGHjBLI", save_dir="data/wavs")
