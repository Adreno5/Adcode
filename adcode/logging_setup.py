"""Where the logs go.

AdCode draws the whole screen, so anything written to stdout or stderr lands
*on top of* the interface and shreds it.  Python's default logging does
exactly that: a warning from the sdk (or from anywhere else) is printed to
stderr and the frame is gone.

So the logs go to a file, and the terminal is left to the terminal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

#: The libraries whose chatter must never reach the terminal.
NOISY_LOGGERS = ('harness', 'infra', 'adcode', 'httpx', 'httpcore',
                 'openai', 'urllib3')


def log_file() -> Path:
    """The file the logs are written to.

    The state directory, never the project: a log next to the code you
    are working on is one more file in your repository.
    """
    from adcode.paths import log_file as _log_file
    return _log_file()


def configure_logging(level: int = logging.WARNING) -> Path:
    """Send every log record to the log file; keep the terminal clean.

    Returns the file the logs go to.  Calling this twice is harmless.
    """
    target = log_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return target
    root = logging.getLogger()
    # A handler that was installed before (or by a library) must not keep
    # writing to the terminal -- not on the root logger, and not on any
    # logger that was given a stream of its own.
    def _is_terminal(handler: logging.Handler) -> bool:
        return (isinstance(handler, logging.StreamHandler)
                and getattr(handler, 'stream', None) in (sys.stderr, sys.stdout))
    for handler in list(root.handlers):
        if _is_terminal(handler):
            root.removeHandler(handler)
    for name in NOISY_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers = [handler for handler in logger.handlers
                           if not _is_terminal(handler)]
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            logger.handlers = [handler for handler in logger.handlers
                               if not _is_terminal(handler)]
    if not any(isinstance(handler, logging.FileHandler)
               and getattr(handler, 'baseFilename', '') == str(target)
               for handler in root.handlers):
        try:
            handler = logging.FileHandler(target, encoding='utf-8')
        except OSError:
            return target
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s: %(message)s'))
        root.addHandler(handler)
        # A logger with no handler of its own falls back to ``lastResort``,
        # which writes to stderr -- the one path that would still land on the
        # interface.  Point it at the same file.
        logging.lastResort = handler
    root.setLevel(level)
    return target
