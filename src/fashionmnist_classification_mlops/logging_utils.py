import os
import sys

from loguru import logger


def setup_logger(log_dir: str = "logs", level: str = "INFO") -> None:
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()  # remove default stderr logger

    # Console
    logger.add(sys.stdout, level=level)

    # File (rotates so it doesn't grow forever)
    logger.add(
        os.path.join(log_dir, "run.log"),
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
