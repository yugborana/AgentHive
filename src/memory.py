"""Memory and Checkpointing for AgentHive.

This module provides abstractions for persisting conversation history
so agents can remember past interactions across multiple sessions.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
import aiosqlite  # We'll use aiosqlite for async database operations
from pydantic import TypeAdapter

from .messages import Message

# A TypeAdapter for serializing a single message to/from JSON rapidly
single_message_adapter = TypeAdapter(Message)


class MemoryStore(ABC):
    """Abstract base class for all memory providers.
    
    Any custom database (Redis, Postgres, Mongo) can be integrated
    by subclassing this and implementing these two methods.
    """

    @abstractmethod
    async def get_messages(self, session_id: str) -> list[Message]:
        """Fetch all historical messages for a given session."""
        ...

    @abstractmethod
    async def add_messages(self, session_id: str, messages: list[Message]) -> None:
        """Append new messages to the session's history."""
        ...


class SQLiteMemoryStore(MemoryStore):
    """A production-ready, lightweight SQLite memory provider.
    
    Saves conversation history to a local SQLite database file.
    Since SQLite doesn't natively support async, we use 'aiosqlite'.
    """

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path

    async def _init_db(self):
        """Ensure the tables exist before we query."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Create an index to make fetching by session_id lightning fast
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)"
            )
            await db.commit()

    async def get_messages(self, session_id: str) -> list[Message]:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT message_json FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                
        # Parse the raw JSON strings back into our strict TypedDicts
        return [single_message_adapter.validate_json(row[0]) for row in rows]

    async def add_messages(self, session_id: str, messages: list[Message]) -> None:
        if not messages:
            return
            
        await self._init_db()
        
        # Serialize the typed dicts into raw JSON strings
        inserts = [
            (session_id, single_message_adapter.dump_json(m).decode("utf-8"))
            for m in messages
        ]
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                "INSERT INTO messages (session_id, message_json) VALUES (?, ?)",
                inserts
            )
            await db.commit()


class InMemoryStore(MemoryStore):
    """A simple dictionary-based memory provider for testing.
    
    Data is lost when the Python script stops.
    """
    
    def __init__(self):
        self._db: dict[str, list[Message]] = {}

    async def get_messages(self, session_id: str) -> list[Message]:
        return self._db.get(session_id, []).copy()

    async def add_messages(self, session_id: str, messages: list[Message]) -> None:
        if session_id not in self._db:
            self._db[session_id] = []
        self._db[session_id].extend(messages)
