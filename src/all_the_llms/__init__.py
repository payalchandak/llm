"""A unified interface for querying Large Language Models (LLMs) across multiple providers."""

import logging
from .llm import LLM
from .model_router import ModelRouter

__version__ = "0.1.3"
__all__ = ["LLM", "ModelRouter"]

# Configure package logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Prevents logs from propagating if not configured

