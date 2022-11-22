import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.utils.logger import get_logger
from tqdm.contrib.logging import logging_redirect_tqdm

logger = get_logger('temp')
logger.info("hello")
logger.disabled = True
logger.info("good bye")
logger.disabled = False
logger.info("OK")

logger = get_logger("test")
from tqdm import tqdm
import time

with logging_redirect_tqdm():
    for i in tqdm(range(10)):
        logger.info(i)
        time.sleep(0.5)
    