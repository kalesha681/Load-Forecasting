import logging
import sys
import structlog
from typing import Optional

def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """
    Configure structured logging for the application.

    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format (bool): If True, output JSON logs. Otherwise, colored console output.
    """
    
    # Shared processors for both JSON and Console
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_format:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer()
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )

    # Note: If we wanted to intercept standard library logging (e.g. from libraries),
    # we would also configure logging.basicConfig here with a structlog wrapper.
    # For this implementation, we focus on our application logs.

def get_logger(name: Optional[str] = None):
    """
    Get a structured logger.
    """
    return structlog.get_logger(name)
