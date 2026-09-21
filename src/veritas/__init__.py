"""Veritas AI — invariant-first, self-evolving AI Agent runtime.

Author: 晨星
"""
from .contracts import Answer, ErrorFamily, RouteTier
from .pipeline import VeritasApp, build_offline_app

__version__ = "0.1.0"

__all__ = ["Answer", "ErrorFamily", "RouteTier", "VeritasApp", "build_offline_app",
           "__version__"]
