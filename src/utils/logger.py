import logging
import os
from datetime import datetime as dt


__FORMAT__ = '%(asctime)s %(name)s %(levelname)s %(message)s'
__formatter = logging.Formatter(__FORMAT__)
logging.basicConfig(format=__FORMAT__)

__logger_dict__ = {}
__log_file__ = None


def get_logger(name: str):
    if __logger_dict__.get(name) is not None:
        return __logger_dict__.get(name)

    LOG_DIR = './logs/'
    LOG_LEVEL = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    for handler in logging.root.handlers[:]:
        print(type(handler))
        logging.root.removeHandler(handler)

    sh = logging.StreamHandler()
    sh.setFormatter(__formatter)
    sh.setLevel(LOG_LEVEL)
    logger.addHandler(sh)

    global __log_file__
    if __log_file__ is None:
        __log_file__ = f"CRAWLER-{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        f = open(os.path.join(LOG_DIR, __log_file__), 'w')
        f.close()

    fh = logging.FileHandler(os.path.join(LOG_DIR, __log_file__), mode='a', encoding='utf-8')

    fh.setFormatter(__formatter)
    fh.setLevel(LOG_LEVEL)
    logger.addHandler(fh)
    
    __logger_dict__[name] = logger
    return logger