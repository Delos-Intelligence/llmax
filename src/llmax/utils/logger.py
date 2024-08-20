"""Logger module."""

import sys

from loguru import logger

logger.add(
    sys.stderr,
    format="{time} {level} {message}",
    level="INFO",
)
