import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.crawling.outlier_remover import OutlierRemover

o = OutlierRemover()
o.run('./data/wavs/ED/Ed Sheeran - Bad Habits [Official Video]')