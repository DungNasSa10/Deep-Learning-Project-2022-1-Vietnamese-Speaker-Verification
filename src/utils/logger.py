import logging
import os
from datetime import datetime as dt


__FORMAT__ = '%(asctime)s %(name)s %(levelname)s %(message)s'
__formatter = logging.Formatter(__FORMAT__)
logging.basicConfig(format=__FORMAT__)

__logger_dict__ = {}


def get_logger(name: str):
    if __logger_dict__.get(name) is not None:
        return __logger_dict__.get(name)

    LOG_DIR = './logs/'
    LOG_LEVEL = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}-{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"), mode='w')
    fh.setFormatter(__formatter)
    fh.setLevel(LOG_LEVEL)
    logger.addHandler(fh)
    
    __logger_dict__[name] = logger
    return logger