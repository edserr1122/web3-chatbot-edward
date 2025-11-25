"""
Centralized logging configuration for the crypto chatbot.
Ensures every entry point (CLI, API, tests, notebooks) writes to the same log
file and propagates INFO-level events from all src.* modules.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import Optional

_LOGGING_CONFIGURED = False
_FILE_HANDLER: Optional[logging.Handler] = None
_CONSOLE_HANDLER: Optional[logging.Handler] = None


def setup_logging(
    *,
    log_file: Optional[str] = None,
    verbose: bool = False,
    enable_console: bool = False,
) -> Optional[logging.Handler]:
    """
    Configure global logging handlers (idempotent).

    Args:
        log_file: Optional explicit log file path. Defaults to CHATBOT_LOG_FILE env
                  or `chatbot.log` in the current working directory.
        verbose:  If True, console handler logs INFO level, otherwise WARNING.
        enable_console: Whether to attach a console handler. CLI sets this True,
                        headless contexts (API/tests) can keep it False.

    Returns:
        The file handler so callers can flush it when needed (e.g., CLI).
    """
    global _LOGGING_CONFIGURED, _FILE_HANDLER, _CONSOLE_HANDLER

    log_file = log_file or os.getenv("CHATBOT_LOG_FILE", "chatbot.log")
    root_logger = logging.getLogger()

    if not _LOGGING_CONFIGURED:
        root_logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)
        _FILE_HANDLER = file_handler

        if enable_console:
            console_handler = _create_console_handler(verbose=verbose)
            root_logger.addHandler(console_handler)
            _CONSOLE_HANDLER = console_handler

        _LOGGING_CONFIGURED = True

        # Ensure all src.* loggers propagate to the root handler
        for logger_name in [
            "src",
            "src.data_sources",
            "src.analyzers",
            "src.tools",
            "src.agents",
            "interfaces",
        ]:
            child_logger = logging.getLogger(logger_name)
            child_logger.setLevel(logging.INFO)
            child_logger.propagate = True
            child_logger.handlers.clear()

    else:
        # Logging already configured; adjust console handler if needed
        if enable_console and _CONSOLE_HANDLER is None:
            console_handler = _create_console_handler(verbose=verbose)
            root_logger.addHandler(console_handler)
            _CONSOLE_HANDLER = console_handler
        elif _CONSOLE_HANDLER is not None:
            _CONSOLE_HANDLER.setLevel(logging.INFO if verbose else logging.WARNING)

    return _FILE_HANDLER


def _create_console_handler(*, verbose: bool) -> logging.Handler:
    """Create a UTF-8 friendly console handler (especially for Windows)."""
    # Windows' default stdout is not UTF-8; wrap if needed
    stream = sys.stdout
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        stream = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )

    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            if verbose
            else "[%(levelname)s] %(message)s"
        )
    )
    return console_handler

