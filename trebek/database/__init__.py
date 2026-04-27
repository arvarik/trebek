"""
Database package — SQLite actor-pattern writer with pipeline operations.

The ``DatabaseWriter`` serializes all database writes through a single-threaded
queue to prevent SQLite concurrency issues. Pipeline-specific query operations
(polling, retries, telemetry) are provided via the ``PipelineQueryMixin``.

Relational data commits (episodes, contestants, clues) are handled by
``operations.commit_episode_to_relational_tables()``.
"""

from trebek.database.writer import DatabaseWriter
from trebek.database.operations import commit_episode_to_relational_tables

__all__ = ["DatabaseWriter", "commit_episode_to_relational_tables"]
