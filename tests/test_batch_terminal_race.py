"""Concurrent terminal-transition tests against a REAL Postgres.

Plan 083 §3.1 (P0): two items of one batch finishing concurrently must leave
the batch in a terminal state with `completed_at` set exactly once. Before the
fix, `_update_item_terminal` recomputed counts *before* taking the parent
`FOR UPDATE` lock, so under READ COMMITTED both finishers could read a stale
non-terminal count and both write `status=running` — a permanently stuck batch.

These tests need a throwaway Postgres and are skipped unless
`EXTRACT_TEST_DATABASE_URL` is set, e.g.:

    docker run -d --name extract-test-pg -e POSTGRES_PASSWORD=test \
        -e POSTGRES_DB=extract_test -p 55432:5432 postgres:16-alpine
    EXTRACT_TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/extract_test \
        uv run pytest tests/test_batch_terminal_race.py
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from extract.core.batch import BatchStatus, ItemStatus
from extract.migrate import apply_migrations
from extract.repos.batches import BatchRepo

TEST_DSN = os.environ.get("EXTRACT_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DSN, reason="EXTRACT_TEST_DATABASE_URL not set (needs a throwaway Postgres)"
)



async def _connect():
    import asyncpg

    return await asyncpg.connect(dsn=TEST_DSN)


async def _seed_running_batch(conn, *, n_items: int) -> tuple[str, list[tuple[str, str]]]:
    """Insert one running batch with `n_items` running, leased items.

    Returns (batch_id, [(item_id, lease_token), ...]).
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    now = datetime.now(tz=UTC).replace(tzinfo=None)
    await conn.execute(
        """
        insert into extract_batches
            (id, customer_id, status, total_items, counts_json, created_at,
             started_at, expires_at)
        values ($1, 'cus_test', $2, $3, $4::jsonb, $5, $5, $6)
        """,
        batch_id,
        BatchStatus.RUNNING,
        n_items,
        json.dumps({"pending": 0, "running": n_items, "succeeded": 0, "failed": 0, "cancelled": 0}),
        now,
        now + timedelta(days=1),
    )
    items: list[tuple[str, str]] = []
    for position in range(n_items):
        item_id = f"item_{uuid.uuid4().hex[:12]}"
        lease = f"lease_{uuid.uuid4().hex[:12]}"
        await conn.execute(
            """
            insert into extract_batch_items
                (id, batch_id, customer_id, file_id, position, status,
                 lease_token, lease_expires_at, attempts, started_at, updated_at)
            values ($1, $2, 'cus_test', null, $3, $4, $5, $6, 1, $7, $7)
            """,
            item_id,
            batch_id,
            position,
            ItemStatus.RUNNING,
            lease,
            now + timedelta(minutes=5),
            now,
        )
        items.append((item_id, lease))
    return batch_id, items


@pytest.fixture()
async def pg():
    conn = await _connect()
    # Apply the REAL migrations so schema drift between migrations/ and the
    # repo fails here first.
    await conn.execute(
        "drop table if exists webhook_deliveries, webhook_events, webhook_endpoints,"
        " extract_batch_items, extract_batches, extract_files, schema_migrations cascade"
    )
    await apply_migrations(TEST_DSN)
    yield conn
    await conn.execute("truncate extract_batches, extract_batch_items cascade")
    await conn.close()


@pytest.fixture()
async def repo():
    r = BatchRepo(dsn=TEST_DSN)
    yield r
    await r.aclose()


@pytest.mark.asyncio
async def test_concurrent_finishers_settle_terminal(pg, repo):
    """The adversarial interleaving: both finishers update their item row
    BEFORE either can take the parent lock (a third connection pins it).
    The last committer must still derive the true terminal status."""
    batch_id, items = await _seed_running_batch(pg, n_items=2)
    (item1, lease1), (item2, lease2) = items

    blocker = await _connect()
    tx = blocker.transaction()
    await tx.start()
    await blocker.execute(
        "select 1 from extract_batches where id = $1 for update", batch_id
    )

    async def finish(item_id: str, lease: str):
        return await repo.update_item_succeeded(
            item_id=item_id,
            lease_token=lease,
            page_count=1,
            result_s3_bucket="results",
            result_s3_key=f"{batch_id}/{item_id}.json",
        )

    t1 = asyncio.create_task(finish(item1, lease1))
    t2 = asyncio.create_task(finish(item2, lease2))
    # Both tasks update their item row, then queue on the parent lock.
    await asyncio.sleep(0.8)
    assert not t1.done() and not t2.done(), "finishers should be blocked on the parent lock"
    await tx.rollback()
    await blocker.close()

    rows = await asyncio.gather(t1, t2)
    assert all(r is not None for r in rows)

    batch = await pg.fetchrow("select * from extract_batches where id = $1", batch_id)
    assert batch["status"] == BatchStatus.COMPLETED
    assert batch["completed_at"] is not None
    counts = json.loads(batch["counts_json"])
    assert counts["succeeded"] == 2
    assert counts["running"] == 0
    # Exactly one of the two returned rows carries the terminal transition
    # (completed_at set); the first committer saw the sibling still running.
    terminal_views = [r for r in rows if r.status == BatchStatus.COMPLETED]
    assert len(terminal_views) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("round_", range(5))
async def test_concurrent_finishers_unorchestrated(pg, repo, round_):
    """Free-running repetition (no lock choreography) — must always settle."""
    batch_id, items = await _seed_running_batch(pg, n_items=3)

    async def finish(item_id: str, lease: str):
        return await repo.update_item_succeeded(
            item_id=item_id,
            lease_token=lease,
            page_count=1,
            result_s3_bucket="results",
            result_s3_key=f"{batch_id}/{item_id}.json",
        )

    await asyncio.gather(*(finish(i, tok) for i, tok in items))
    batch = await pg.fetchrow("select * from extract_batches where id = $1", batch_id)
    assert batch["status"] == BatchStatus.COMPLETED
    assert batch["completed_at"] is not None


@pytest.mark.asyncio
async def test_mixed_outcome_derives_partially_failed(pg, repo):
    batch_id, items = await _seed_running_batch(pg, n_items=2)
    (item1, lease1), (item2, lease2) = items
    await repo.update_item_succeeded(
        item_id=item1,
        lease_token=lease1,
        page_count=1,
        result_s3_bucket="results",
        result_s3_key="k1",
    )
    await repo.update_item_failed(
        item_id=item2,
        lease_token=lease2,
        error_code="internal_error",
        error_message="boom",
    )
    batch = await pg.fetchrow("select * from extract_batches where id = $1", batch_id)
    assert batch["status"] == BatchStatus.PARTIALLY_FAILED
    assert batch["completed_at"] is not None


@pytest.mark.asyncio
async def test_cancel_batch_sets_completed_at_once(pg, repo):
    """cancel_batch on an all-pending batch goes terminal with completed_at
    set; the null-guard keeps completed_at stable on any later write."""
    batch_id, items = await _seed_running_batch(pg, n_items=2)
    # Flip both items back to pending (cancel only touches pending items).
    for item_id, _ in items:
        await pg.execute(
            "update extract_batch_items set status = $2, lease_token = null where id = $1",
            item_id,
            ItemStatus.PENDING,
        )
    row = await repo.cancel_batch(batch_id=batch_id, customer_id="cus_test")
    assert row is not None
    assert row.status == BatchStatus.CANCELLED
    first_completed_at = row.completed_at
    assert first_completed_at is not None
    # Idempotent second cancel: early-return, timestamp unchanged.
    row2 = await repo.cancel_batch(batch_id=batch_id, customer_id="cus_test")
    assert row2 is not None
    assert row2.completed_at == first_completed_at
