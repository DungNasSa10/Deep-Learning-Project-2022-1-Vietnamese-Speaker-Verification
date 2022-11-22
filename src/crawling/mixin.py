from utils.logger import get_logger


class StepMixin:
    def __init__(self) -> None:
        self.__logger = get_logger(self.__class__.__name__)
        self.__verbose = True
    
    @property
    def logger(self):
        return self.__logger
    
    @property
    def verbose(self) -> bool:
        return self.__verbose
    
    @verbose.setter
    def verbose(self, _verbose: bool):
        self.__verbose = _verbose
        self.__logger.disabled = not _verbose

    def run(self, *args, **kwargs):
        raise NotImplementedError()