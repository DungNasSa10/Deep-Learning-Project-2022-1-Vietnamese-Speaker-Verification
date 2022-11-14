import os
import sys

sys.path.append(os.getcwd())

from src.crawling.outlier_remover import OutlierRemover

o = OutlierRemover()
o.run('./data/wavs/ED/Ed Sheeran - Bad Habits [Official Video]')