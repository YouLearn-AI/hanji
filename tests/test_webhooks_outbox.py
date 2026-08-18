"""Outbox enqueue + dispatcher persistence tests against a REAL Postgres.
Applies the actual migrations/ SQL, so schema drift between the migrations
and the Python repo fails here first.

Skipped unless EXTRACT_TEST_DATABASE_URL is set (needs a throwaway
Postgres, e.g. the docker-compose batch profile's on localhost:5433).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from extract.core.batch import BatchStatus, ItemStatus
from extract.core.webhooks import generate_secret
from extract.migrate import apply_migrations
from extract.repos.batches import BatchRepo
from extract.repos.webhooks import WebhookRepo

TEST_DSN = os.environ.get("EXTRACT_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DSN, reason="EXTRACT_TEST_DATABASE_URL not set (needs a throwaway Postgres)"
)



async def _connect():
    import asyncpg

    return await asyncpg.connect(dsn=TEST_DSN)


def _utcnow() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)


@pytest.fixture()
async def pg():
    conn = await _connect()
    await conn.execute(
        "drop table if exists webhook_deliveries, webhook_events, webhook_endpoints,"
        " extract_batch_items, extract_batches, extract_files, schema_migrations cascade"
    )
    await apply_migrations(TEST_DSN)
    yield conn
    await conn.execute(
        "truncate webhook_deliveries, webhook_events, webhook_endpoints,"
        " extract_batches, extract_batch_items cascade"
    )
    await conn.close()


@pytest.fixture()
async def repo():
    r = BatchRepo(dsn=TEST_DSN)
    yield r
    await r.aclose()


@pytest.fixture()
async def whrepo():
    r = WebhookRepo(dsn=TEST_DSN)
    yield r
    await r.aclose()


async def _seed_batch(
    pg,
    *,
    n_items: int,
    webhook_mode: str | None,
    webhook_url: str | None = None,
    completed_ago_minutes: int | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """One batch with `n_items` running leased items (or, when
    completed_ago_minutes is set, an already-terminal batch with no items)."""
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    now = _utcnow()
    terminal = completed_ago_minutes is not None
    await pg.execute(
        """
        insert into extract_batches
            (id, customer_id, status, total_items, counts_json, metadata_json,
             webhook_mode, webhook_url, created_at, started_at, completed_at, expires_at)
        values ($1, 'cus_test', $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8, $8, $9, $10)
        """,
        batch_id,
        BatchStatus.COMPLETED if terminal else BatchStatus.RUNNING,
        n_items,
        json.dumps(
            {"pending": 0, "running": 0 if terminal else n_items,
             "succeeded": n_items if terminal else 0, "failed": 0, "cancelled": 0}
        ),
        json.dumps({"prior_auth_id": "PA-1"}),
        webhook_mode,
        webhook_url,
        now,
        (now - timedelta(minutes=completed_ago_minutes)) if terminal else None,
        now + timedelta(days=1),
    )
    items: list[tuple[str, str]] = []
    if not terminal:
        for position in range(n_items):
            item_id = f"item_{uuid.uuid4().hex[:12]}"
            lease = f"lease_{uuid.uuid4().hex[:12]}"
            await pg.execute(
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


async def _register_endpoint(pg, *, enabled: bool = True, url: str | None = None) -> str:
    endpoint_id = f"whe_{uuid.uuid4().hex[:12]}"
    await pg.execute(
        """
        insert into webhook_endpoints
            (id, customer_id, url, secret_ciphertext, secret_key_id, enabled)
        values ($1, 'cus_test', $2, $3, 'local-dev', $4)
        """,
        endpoint_id,
        url or f"https://hooks.example.com/{endpoint_id}",
        # local-dev lane: ciphertext == base64(plaintext)
        __import__("base64").b64encode(generate_secret().encode()).decode(),
        enabled,
    )
    return endpoint_id


async def _finish_all(repo: BatchRepo, batch_id: str, items: list[tuple[str, str]]):
    for item_id, lease in items:
        await repo.update_item_succeeded(
            item_id=item_id,
            lease_token=lease,
            page_count=1,
            result_s3_bucket="results",
            result_s3_key=f"{batch_id}/{item_id}.json",
        )


# --- Enqueue semantics ---------------------------------------------------------


async def test_opt_in_default_no_event_even_with_endpoints(pg, repo):
    await _register_endpoint(pg)
    batch_id, items = await _seed_batch(pg, n_items=1, webhook_mode=None)
    await _finish_all(repo, batch_id, items)
    n = await pg.fetchval("select count(*) from webhook_events where batch_id = $1", batch_id)
    assert n == 0


async def test_disabled_mode_no_event(pg, repo):
    await _register_endpoint(pg)
    batch_id, items = await _seed_batch(pg, n_items=1, webhook_mode="disabled")
    await _finish_all(repo, batch_id, items)
    n = await pg.fetchval("select count(*) from webhook_events where batch_id = $1", batch_id)
    assert n == 0


async def test_svix_mode_enqueues_one_event_fanout_per_endpoint(pg, repo):
    ep1 = await _register_endpoint(pg)
    ep2 = await _register_endpoint(pg)
    await _register_endpoint(pg, enabled=False)  # disabled: no delivery
    batch_id, items = await _seed_batch(pg, n_items=2, webhook_mode="svix")
    await asyncio.gather(
        *(
            repo.update_item_succeeded(
                item_id=i, lease_token=t, page_count=1,
                result_s3_bucket="results", result_s3_key=f"{batch_id}/{i}.json",
            )
            for i, t in items
        )
    )
    events = await pg.fetch("select * from webhook_events where batch_id = $1", batch_id)
    assert len(events) == 1  # exactly-once under concurrent finishers
    body = json.loads(events[0]["body"])
    assert body["type"] == "batch.update"
    assert body["status"] == "completed"
    assert body["total_items"] == 2
    assert body["metadata"] == {"prior_auth_id": "PA-1"}
    assert body["completed_at"].endswith("Z")
    # The result-retention deadline is surfaced so consumers know their fetch
    # window (mirrors the batch's expires_at / the resend TTL bound).
    assert body["results_expires_at"].endswith("Z")
    deliveries = await pg.fetch(
        "select * from webhook_deliveries where event_id = $1", events[0]["id"]
    )
    assert {d["endpoint_id"] for d in deliveries} == {ep1, ep2}
    assert all(d["signed"] for d in deliveries)


async def test_svix_mode_without_endpoints_skips(pg, repo):
    batch_id, items = await _seed_batch(pg, n_items=1, webhook_mode="svix")
    await _finish_all(repo, batch_id, items)
    n = await pg.fetchval("select count(*) from webhook_events where batch_id = $1", batch_id)
    assert n == 0


async def test_direct_mode_single_unsigned_delivery(pg, repo):
    batch_id, items = await _seed_batch(
        pg, n_items=1, webhook_mode="direct", webhook_url="https://hooks.example.com/direct"
    )
    await _finish_all(repo, batch_id, items)
    row = await pg.fetchrow(
        """
        select d.* from webhook_deliveries d
          join webhook_events e on e.id = d.event_id
         where e.batch_id = $1
        """,
        batch_id,
    )
    assert row is not None
    assert row["endpoint_id"] is None
    assert not row["signed"]
    assert row["url"] == "https://hooks.example.com/direct"


async def test_cancel_batch_enqueues_exactly_once(pg, repo):
    await _register_endpoint(pg)
    batch_id, items = await _seed_batch(pg, n_items=2, webhook_mode="svix")
    for item_id, _ in items:
        await pg.execute(
            "update extract_batch_items set status = $2, lease_token = null where id = $1",
            item_id,
            ItemStatus.PENDING,
        )
    await repo.cancel_batch(batch_id=batch_id, customer_id="cus_test")
    await repo.cancel_batch(batch_id=batch_id, customer_id="cus_test")  # idempotent
    events = await pg.fetch("select * from webhook_events where batch_id = $1", batch_id)
    assert len(events) == 1
    assert json.loads(events[0]["body"])["status"] == "cancelled"


async def test_partially_failed_body(pg, repo):
    await _register_endpoint(pg)
    batch_id, items = await _seed_batch(pg, n_items=2, webhook_mode="svix")
    (i1, t1), (i2, t2) = items
    await repo.update_item_succeeded(
        item_id=i1, lease_token=t1, page_count=1, result_s3_bucket="r", result_s3_key="k"
    )
    await repo.update_item_failed(
        item_id=i2, lease_token=t2, error_code="internal_error", error_message="boom"
    )
    event = await pg.fetchrow("select * from webhook_events where batch_id = $1", batch_id)
    body = json.loads(event["body"])
    assert body["status"] == "partially_failed"
    assert body["counts"] == {"pending": 0, "running": 0, "succeeded": 1, "failed": 1, "cancelled": 0}


# --- Reconciler ---------------------------------------------------------------


async def test_reconciler_bounded_and_mode_aware(pg, whrepo):
    await _register_endpoint(pg)
    # Eligible: svix batch, terminal 30 min ago, no event.
    eligible, _ = await _seed_batch(pg, n_items=1, webhook_mode="svix", completed_ago_minutes=30)
    # Not eligible: NULL mode (never retroactive), disabled, inside grace,
    # outside window W.
    null_mode, _ = await _seed_batch(pg, n_items=1, webhook_mode=None, completed_ago_minutes=30)
    disabled, _ = await _seed_batch(
        pg, n_items=1, webhook_mode="disabled", completed_ago_minutes=30
    )
    fresh, _ = await _seed_batch(pg, n_items=1, webhook_mode="svix", completed_ago_minutes=2)
    ancient, _ = await _seed_batch(
        pg, n_items=1, webhook_mode="svix", completed_ago_minutes=60 * 48
    )
    # Eligible: direct-mode batch (no registered endpoint required).
    direct, _ = await _seed_batch(
        pg,
        n_items=1,
        webhook_mode="direct",
        webhook_url="https://hooks.example.com/direct",
        completed_ago_minutes=30,
    )
    reconciled = await whrepo.reconcile_missing_events()
    assert reconciled == 2
    got = {
        r["batch_id"]
        for r in await pg.fetch("select batch_id from webhook_events where batch_id is not null")
    }
    assert got == {eligible, direct}
    assert null_mode not in got and disabled not in got
    assert fresh not in got and ancient not in got
    # Idempotent second sweep.
    assert await whrepo.reconcile_missing_events() == 0


# --- Dispatcher persistence ------------------------------------------------------


async def _one_pending_delivery(pg, repo) -> str:
    await _register_endpoint(pg)
    batch_id, items = await _seed_batch(pg, n_items=1, webhook_mode="svix")
    await _finish_all(repo, batch_id, items)
    return await pg.fetchval(
        """
        select d.id from webhook_deliveries d
          join webhook_events e on e.id = d.event_id
         where e.batch_id = $1
        """,
        batch_id,
    )


async def test_claim_lease_and_ladder(pg, repo, whrepo):
    delivery_id = await _one_pending_delivery(pg, repo)
    claimed = await whrepo.claim_next_delivery(lease_seconds=60)
    assert claimed is not None and claimed.delivery_id == delivery_id
    assert claimed.signed and claimed.endpoint_enabled
    assert json.loads(claimed.body)["type"] == "batch.update"
    # Nothing else claimable while leased.
    assert await whrepo.claim_next_delivery(lease_seconds=60) is None
    # Failed attempt 1 → pending again 5s out.
    status = await whrepo.mark_attempt_failed(
        delivery_id=delivery_id, lease_token=claimed.lease_token, status_code=500, error="HTTP 500"
    )
    assert status == "pending"
    row = await pg.fetchrow("select * from webhook_deliveries where id = $1", delivery_id)
    assert row["attempts"] == 1 and row["last_status_code"] == 500
    # Stale lease token can no longer complete it.
    assert not await whrepo.mark_succeeded(
        delivery_id=delivery_id, lease_token=claimed.lease_token, status_code=200
    )


async def test_expired_delivering_lease_is_reclaimed(pg, repo, whrepo):
    delivery_id = await _one_pending_delivery(pg, repo)
    claimed = await whrepo.claim_next_delivery(lease_seconds=60)
    assert claimed is not None
    # Simulate a dispatcher crash: force the lease into the past.
    await pg.execute(
        "update webhook_deliveries set lease_expires_at = now() - interval '1 minute' where id = $1",
        delivery_id,
    )
    reclaimed = await whrepo.claim_next_delivery(lease_seconds=60)
    assert reclaimed is not None and reclaimed.delivery_id == delivery_id
    assert reclaimed.lease_token != claimed.lease_token


async def test_ladder_exhaustion_marks_failed(pg, repo, whrepo):
    delivery_id = await _one_pending_delivery(pg, repo)
    for n in range(1, 9):
        await pg.execute(
            "update webhook_deliveries set next_attempt_at = now() - interval '1 second' where id = $1",
            delivery_id,
        )
        claimed = await whrepo.claim_next_delivery(lease_seconds=60)
        assert claimed is not None, f"attempt {n} not claimable"
        status = await whrepo.mark_attempt_failed(
            delivery_id=delivery_id,
            lease_token=claimed.lease_token,
            status_code=503,
            error="HTTP 503",
        )
    assert status == "failed"
    row = await pg.fetchrow("select * from webhook_deliveries where id = $1", delivery_id)
    assert row["attempts"] == 8


async def test_kill_switch_disabled_endpoint_visible_at_claim(pg, repo, whrepo):
    delivery_id = await _one_pending_delivery(pg, repo)
    await pg.execute(
        """
        update webhook_endpoints set enabled = false
         where id = (select endpoint_id from webhook_deliveries where id = $1)
        """,
        delivery_id,
    )
    claimed = await whrepo.claim_next_delivery(lease_seconds=60)
    assert claimed is not None
    assert claimed.endpoint_enabled is False
    assert await whrepo.mark_cancelled(
        delivery_id=delivery_id, lease_token=claimed.lease_token
    )


async def test_oldest_pending_age_gauge(pg, repo, whrepo):
    assert await whrepo.oldest_pending_age_seconds() is None
    await _one_pending_delivery(pg, repo)
    age = await whrepo.oldest_pending_age_seconds()
    assert age is not None and age >= 0.0
