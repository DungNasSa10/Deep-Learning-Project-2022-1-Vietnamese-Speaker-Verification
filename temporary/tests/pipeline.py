import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.crawling.pipeline import Pipeline

p = Pipeline()

# p.run('./data/Voice list - Long.csv')
p.run('https://drive.google.com/file/d/1BL4tkLkPPDeuYwlCaMvIrKYCst8E4zEN/view?usp=share_link')