"""
db.py — Database session setup.

Defaults to a local SQLite file (zero-config for development). Set
DATABASE_URL to a Postgres DSN (e.g. postgresql+psycopg://...) for
production — SQLAlchemy's engine construction is identical either way,
so no application code changes when swapping databases.
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./mindai.db")

# check_same_thread=False is required for SQLite when accessed from
# FastAPI's threaded request handlers; irrelevant (ignored) for Postgres.
# timeout=30 makes a writer that finds SQLite briefly locked (e.g. two
# short writes landing at nearly the same moment from different threadpool
# threads) wait up to 30s and retry rather than immediately raising
# "database is locked" — a real failure mode hit during testing.
connect_args = {"check_same_thread": False, "timeout": 30} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (no-op for ones that already exist), then patch
    in any columns added to the models after a table was first created.
    `create_all` never alters existing tables, so a genuine migration tool
    (Alembic) is the correct long-term answer — this is a minimal stopgap
    scoped to local SQLite dev databases, not a general migration system.
    """
    import backend.models  # noqa: F401  (ensure models are registered on Base.metadata)

    Base.metadata.create_all(bind=engine)
    _patch_missing_columns()


def _patch_missing_columns() -> None:
    if not DATABASE_URL.startswith("sqlite"):
        return  # Postgres/other targets: use a real migration tool instead.

    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                default_clause = ""
                if column.default is not None and column.default.is_scalar:
                    default_clause = f" DEFAULT {column.default.arg!r}"
                conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {col_type}{default_clause}"))
