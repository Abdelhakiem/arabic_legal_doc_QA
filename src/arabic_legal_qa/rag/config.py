"""Backward-compatible imports for centralized project settings.

New code should import configuration objects and settings from
``helpers.config``. This module remains temporarily so earlier notebooks and
external callers do not break during the configuration migration.
"""
from helpers.config import (
    DEFAULT_EMBEDDING_LIMIT,
    DEFAULT_EMBEDDING_MODEL,
    EXPECTED_ARTICLES,
    EXTRACTOR_VERSION,
    REPEAL_RANGES,
    EmbeddingConfig,
    Settings,
    get_settings,
    model_cache_dir,
    project_root,
)

__all__ = [
    "EmbeddingConfig",
    "Settings",
    "get_settings",
    "model_cache_dir",
    "project_root",
    "EXTRACTOR_VERSION",
    "EXPECTED_ARTICLES",
    "REPEAL_RANGES",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_LIMIT",
]
