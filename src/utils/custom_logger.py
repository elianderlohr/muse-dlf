import logging
from accelerate import Accelerator


class CustomLogger:
    def __init__(self, logger, accelerate_used):
        self.logger = logger
        self.accelerate_used = accelerate_used
        if self.accelerate_used:
            self.accelerator = Accelerator()

    def log(self, level, msg, *args, **kwargs):
        main_process_only = kwargs.pop("main_process_only", False)

        if self.accelerate_used:
            if main_process_only and not self.accelerator.is_main_process:
                return

            rank = self.accelerator.process_index
            formatted_msg = f"Rank {rank}: {msg}"
        else:
            formatted_msg = msg

        if hasattr(self.logger, level):
            method = getattr(self.logger, level)
            method(formatted_msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log("debug", msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log("info", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log("warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log("error", msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log("critical", msg, *args, **kwargs)

    def set_log_level(self, log_level):
        self.logger.setLevel(getattr(logging, log_level.upper()))
