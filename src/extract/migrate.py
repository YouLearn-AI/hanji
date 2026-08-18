"""Apply the SQL migrations in ``migrations/`` in order.

Usage::

    python -m extract.migrate            # apply pending migrations
    python -m extract.migrate --check    # exit 1 if any migration is pending

Idempotent: applied filenames are recorded in ``schema_migrations`` and each
file runs inside one transaction, so a crash mid-file leaves nothing half
applied. Files are ordered lexicographically (``0001_…``, ``0002_…``); an
already-applied file is never re-run, so treat committed migrations as
immutable and add a new file instead of editing one.

Reads ``DATABASE_URL`` from the environment / ``.env`` (see
``extract.config``). The docker-compose entrypoints run this before booting
the API and worker.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from extract.config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    return sorted(p for p in directory.glob("*.sql") if p.is_file())


async def apply_migrations(
    dsn: str,
    *,
    directory: Path = MIGRATIONS_DIR,
    check_only: bool = False,
) -> list[str]:
    """Apply pending migrations; return the filenames applied (or, with
    ``check_only``, the filenames that WOULD be applied)."""
    import asyncpg

    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            """
            create table if not exists schema_migrations (
                filename text primary key,
                applied_at timestamp not null default now()
            )
            """
        )
        applied = {
            r["filename"] for r in await conn.fetch("select filename from schema_migrations")
        }
        pending = [p for p in migration_files(directory) if p.name not in applied]
        if check_only:
            return [p.name for p in pending]
        ran: list[str] = []
        for path in pending:
            async with conn.transaction():
                await conn.execute(path.read_text())
                await conn.execute(
                    "insert into schema_migrations (filename) values ($1)", path.name
                )
            ran.append(path.name)
        return ran
    finally:
        await conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if any migration is pending"
    )
    args = parser.parse_args(argv)
    dsn = settings.DATABASE_URL
    if not dsn:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 2
    result = asyncio.run(apply_migrations(dsn, check_only=args.check))
    if args.check:
        if result:
            print("pending migrations:", ", ".join(result))
            return 1
        print("migrations up to date")
        return 0
    for name in result:
        print(f"applied {name}")
    if not result:
        print("migrations up to date")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())
