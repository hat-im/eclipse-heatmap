"""Logging setup."""

from __future__ import annotations

import logging

from tqdm import tqdm


class _TqdmLoggingHandler(logging.Handler):
    """Routes log records through tqdm.write() so they don't garble an active progress bar.

    logging's default StreamHandler and tqdm's bar both write directly to
    stderr with no coordination -- interleaved, they corrupt each other's
    output. tqdm.write() clears the bar, prints the line, then redraws it.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(level: str = "INFO") -> logging.Logger:
    handler = _TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    logger = logging.getLogger("eclipse_heatmap")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
