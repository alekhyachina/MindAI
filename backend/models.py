"""
models.py — SQLAlchemy ORM models for users, ingested repos, and chat history.

Kept deliberately small: this is the persistence layer needed for the web UI
(accounts + conversation history), not a general-purpose schema. The actual
vector data (code chunks, embeddings) lives in Qdrant, not here — `Repo`
only stores the metadata needed to reopen an existing Qdrant collection.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


def generate_id() -> str:
    """
    Public so callers can generate a row's id BEFORE constructing/inserting
    it (e.g. main.py needs Repo.id to derive a Qdrant collection name prior
    to running ingestion, which can take minutes — see the ingest_repo
    docstring). SQLAlchemy's `default=` on the column only fires at
    flush/insert time, not at object construction, so reading `.id` off a
    freshly-constructed-but-not-yet-flushed row returns None, not a UUID.
    """
    return str(uuid.uuid4())


def _uuid() -> str:
    return generate_id()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    repos: Mapped[list["Repo"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class Repo(Base):
    """
    A GitHub repository being (or having been) ingested, scoped to its
    owning user. `status` tracks the background ingestion job — see
    backend/main.py's ingest_repo / _run_ingestion for the state machine:
    pending -> ingesting -> ready, or -> failed with error_message set.
    """

    __tablename__ = "repos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    github_url: Mapped[str] = mapped_column(String(500), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    qdrant_collection_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    total_chunks: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | ingesting | ready | failed
    # Typed as Mapped[str] (not Mapped[Optional[str]]) despite being
    # nullable=True — SQLAlchemy 2.0.36 under Python 3.14 fails to resolve
    # stringified `Optional[...]`/`X | None` annotations at mapper-config
    # time (TypeError in sqlalchemy.util.typing.make_union_type). The
    # explicit nullable=True on the column is what actually controls
    # nullability; the annotation here is just for editor/type-checker
    # hints and deliberately avoids the broken union-inference path.
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    owner: Mapped["User"] = relationship(back_populates="repos")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="repo", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    repo_id: Mapped[str] = mapped_column(String(36), ForeignKey("repos.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    owner: Mapped["User"] = relationship(back_populates="conversations")
    repo: Mapped["Repo"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    refused: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
