import os
import sys

sys.path.append(os.getcwd())

from src.crawling.pipeline import Pipeline

p = Pipeline()

p.run('./data/Voice list - Long.csv')