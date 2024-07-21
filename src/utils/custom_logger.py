class CustomLogger:
    def __init__(self, logger, accelerate_used):
        self.logger = logger
        self.accelerate_used = accelerate_used

    def log(self, level, msg, *args, **kwargs):
        if self.accelerate_used and "main_process_only" in kwargs:
            main_process_only = kwargs.pop("main_process_only")
            if hasattr(self.logger, level):
                method = getattr(self.logger, level)
                method(msg, *args, main_process_only=main_process_only)
        else:
            if hasattr(self.logger, level):
                # remove main_process_only from kwargs if it exists
                kwargs.pop("main_process_only", None)

                method = getattr(self.logger, level)
                method(msg, *args, **kwargs)

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
