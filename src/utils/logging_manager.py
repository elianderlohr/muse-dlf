import logging


class LoggerManager:
    _instance = None
    _accelerate_used = False
    _log_level = "INFO"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def use_accelerate(cls, accelerate_used, log_level="INFO"):
        cls._accelerate_used = accelerate_used
        cls._log_level = log_level

    @classmethod
    def get_logger(cls, name):
        if cls._accelerate_used:
            try:
                from accelerate.logging import get_logger as get_accelerate_logger

                logger = get_accelerate_logger(name)
                logger.setLevel(
                    getattr(logging, cls._log_level)
                )  # Set the logging level
                return logger
            except ImportError:
                pass  # Fall through to standard logger if Accelerate is not available

        # Setup standard logger
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(getattr(logging, cls._log_level))
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, cls._log_level))
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger
