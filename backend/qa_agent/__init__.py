"""
QA Agent package for Paper Search Engine.
Provides single-paper and multi-paper question answering capabilities.
"""

from .agent import QAAgent
from .config import qa_config

__all__ = ["QAAgent", "qa_config"]
