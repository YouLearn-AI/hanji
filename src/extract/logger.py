import contextlib
import logging
import os

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"


class CustomFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: PURPLE,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, "")
        levelname = record.levelname
        colored = f"{color}{levelname}{RESET}:"
        record.levelname = f"{colored:<18}"
        with contextlib.suppress(ValueError):
            record.filename = os.path.relpath(record.pathname)
        return super().format(record)


def get_logger(name: str = "extract", level: int | None = None) -> logging.Logger:
    level = level if level is not None else int(os.environ.get("EXTRACT_LOG_LEVEL", logging.INFO))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            CustomFormatter("{levelname} {filename}:{lineno} {funcName} -> {message}", style="{")
        )
        logger.addHandler(handler)

    return logger
