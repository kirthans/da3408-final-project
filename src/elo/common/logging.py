"""Course-format logging with a request ID from the current context."""

import logging
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="-")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s [request_id=%(request_id)s]"


class RequestIdFilter(logging.Filter):
    """Attach context to each record at emission time."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Make request_id available to the formatter."""
        record.request_id = request_id.get()
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger without adding duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(RequestIdFilter())
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
