"""Structured (one-JSON-object-per-line) request logging to stdout, the
standard place a containerized service's logs go (picked up by `docker logs`
without any extra log-shipping setup)."""
import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "logger": record.name, "message": record.getMessage()}
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload)


def get_request_logger() -> logging.Logger:
    logger = logging.getLogger("ragpipeline.serving")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_request(logger: logging.Logger, **fields) -> None:
    logger.info("query", extra={"extra_fields": fields})
