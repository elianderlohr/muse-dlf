import logging
from custom_logger import CustomLogger


class LoggerManager:
    _instance = None
    _accelerate_used = False
    _log_level = "INFO"
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def use_accelerate(cls, accelerate_used, log_level="INFO"):
        cls._accelerate_used = accelerate_used
        cls._log_level = log_level
        logging.basicConfig(
            level=getattr(logging, cls._log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @classmethod
    def _configure_accelerate_logger(cls, name):
        from accelerate.logging import get_logger as get_accelerate_logger

        logger = get_accelerate_logger(name)
        return CustomLogger(logger, cls._accelerate_used)

    @classmethod
    def _configure_standard_logger(cls, name):
        logger = logging.getLogger(name)
        return CustomLogger(logger, cls._accelerate_used)

    @classmethod
    def get_logger(cls, name):
        if name in cls._loggers:
            return cls._loggers[name]

        if cls._accelerate_used:
            try:
                logger = cls._configure_accelerate_logger(name)
                cls._loggers[name] = logger
                return logger
            except ImportError:
                print("Accelerate not available")
                pass  # Fall through to standard logger if Accelerate is not available

        logger = cls._configure_standard_logger(name)
        cls._loggers[name] = logger
        return logger

    @classmethod
    def clear(cls):
        cls._loggers.clear()
