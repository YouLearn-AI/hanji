"""Dispatcher e2e against real Postgres (plan 083 §5): worker completes a
batch → signed batch.update delivered → verified with the REAL svix lib →
dedup key stable across retries → kill switch → log-field policy sweep.

The network edge is a captured httpx.MockTransport (the SSRF/pinning layer
has its own suite in test_webhooks_transport.py); everything else — enqueue,
claim, signing, ladder, lease guards, logging — is the production code path.

Skipped unless EXTRACT_TEST_DATABASE_URL is set.
"""

# ruff: noqa: F811 — the imported `pg` fixture is intentionally shadowed by
# pytest fixture parameters.

from __future__ import annotations

import json
import logging
import os

import httpx
import pytest
from svix.webhooks import Webhook

from extract.clients import webhook_transport
from extract.clients.webhook_secrets import WebhookSecretCrypto
from extract.config import settings
from extract.repos.batches import BatchRepo
from extract.repos.webhooks import WebhookRepo
from extract.workers.webhook_dispatcher import WebhookDispatcher
from tests.test_webhooks_outbox import (  # noqa: F401 — pg fixture reused
    _finish_all,
    _register_endpoint,
    _seed_batch,
    pg,
)

TEST_DSN = os.environ.get("EXTRACT_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DSN, reason="EXTRACT_TEST_DATABASE_URL not set (needs a throwaway Postgres)"
)

FORBIDDEN_LOG_SUBSTRINGS = (
    "hooks.example.com",  # endpoint URL host
    "/hook-path-secret",  # endpoint URL path
    "whsec_",  # signing secret material
    "PA-1",  # customer metadata value
    "prior_auth_id",  # customer metadata key
)


class _Receiver:
    """Captures deliveries; scriptable per-attempt status codes."""

    def __init__(self, statuses: list[int] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._statuses = statuses or []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self._statuses.pop(0) if self._statuses else 200
        return httpx.Response(status)


@pytest.fixture()
def dispatcher_factory(monkeypatch):

    async def fake_resolve(host, port):
        return "93.184.216.34"

    monkeypatch.setattr(webhook_transport, "_resolve_public_ip", fake_resolve)

    def make(receiver: _Receiver) -> WebhookDispatcher:
        d = WebhookDispatcher(
            repo=WebhookRepo(dsn=TEST_DSN),
            crypto=WebhookSecretCrypto(),
        )
        # Swap the network edge for the captured transport.
        d._client = httpx.AsyncClient(
            transport=httpx.MockTransport(receiver), follow_redirects=False
        )
        return d

    return make


@pytest.fixture()
async def repo():
    r = BatchRepo(dsn=TEST_DSN)
    yield r
    await r.aclose()


async def _drain(dispatcher: WebhookDispatcher, *, rounds: int = 10) -> None:
    """Run claim→deliver until the queue is empty (bounded)."""
    for _ in range(rounds):
        claimed = await dispatcher.repo.claim_next_delivery(
            lease_seconds=settings.EXTRACT_WEBHOOK_LEASE_SECONDS
        )
        if claimed is None:
            return
        await dispatcher._deliver_one(claimed)


async def test_e2e_signed_delivery_verifies_and_dedups(pg, repo, dispatcher_factory, caplog):
    endpoint_id = await _register_endpoint(pg, url="https://hooks.example.com/hook-path-secret")
    secret_ciphertext = await pg.fetchval(
        "select secret_ciphertext from webhook_endpoints where id = $1", endpoint_id
    )
    secret = __import__("base64").b64decode(secret_ciphertext).decode()

    batch_id, items = await _seed_batch(pg, n_items=2, webhook_mode="svix")
    with caplog.at_level(logging.INFO):
        await _finish_all(repo, batch_id, items)
        receiver = _Receiver(statuses=[500, 200])  # first attempt fails → retry
        dispatcher = dispatcher_factory(receiver)
        try:
            await _drain(dispatcher)
            # Force the retry due and drain again.
            await pg.execute(
                "update webhook_deliveries set next_attempt_at = now() - interval '1 second'"
                " where status = 'pending'"
            )
            await _drain(dispatcher)
        finally:
            await dispatcher.stop()

    assert len(receiver.requests) == 2
    first, second = receiver.requests
    # The svix-id (dedup key) is stable across attempts; signatures differ
    # (fresh timestamp), the body bytes are identical.
    assert first.headers["svix-id"] == second.headers["svix-id"]
    assert first.content == second.content
    # The REAL svix library verifies the delivered attempt.
    verified = Webhook(secret).verify(second.content, dict(second.headers))
    assert verified["batch_id"] == batch_id
    assert verified["status"] == "completed"
    assert verified["metadata"] == {"prior_auth_id": "PA-1"}
    # Delivery row settled.
    row = await pg.fetchrow(
        "select * from webhook_deliveries where endpoint_id = $1", endpoint_id
    )
    assert row["status"] == "succeeded"
    assert row["attempts"] == 2

    # --- §3.8 log-field policy sweep over the whole lifecycle -------------
    for record in caplog.records:
        line = record.getMessage() + json.dumps(
            {k: str(v) for k, v in record.__dict__.items()}, default=str
        )
        for forbidden in FORBIDDEN_LOG_SUBSTRINGS:
            assert forbidden not in line, (
                f"forbidden content {forbidden!r} leaked into log event {record.getMessage()!r}"
            )


async def test_e2e_direct_mode_unsigned(pg, repo, dispatcher_factory):
    batch_id, items = await _seed_batch(
        pg, n_items=1, webhook_mode="direct", webhook_url="https://hooks.example.com/direct"
    )
    await _finish_all(repo, batch_id, items)
    receiver = _Receiver()
    dispatcher = dispatcher_factory(receiver)
    try:
        await _drain(dispatcher)
    finally:
        await dispatcher.stop()
    assert len(receiver.requests) == 1
    request = receiver.requests[0]
    assert "svix-signature" not in request.headers  # unsigned (Reducto parity)
    body = json.loads(request.content)
    assert body["batch_id"] == batch_id
    assert body["metadata"] == {"prior_auth_id": "PA-1"}  # secret-in-metadata intact


async def test_e2e_kill_switch_cancels_in_flight_retries(pg, repo, dispatcher_factory):
    endpoint_id = await _register_endpoint(pg)
    batch_id, items = await _seed_batch(pg, n_items=1, webhook_mode="svix")
    await _finish_all(repo, batch_id, items)
    await pg.execute("update webhook_endpoints set enabled = false where id = $1", endpoint_id)
    receiver = _Receiver()
    dispatcher = dispatcher_factory(receiver)
    try:
        await _drain(dispatcher)
    finally:
        await dispatcher.stop()
    assert receiver.requests == []  # nothing sent
    row = await pg.fetchrow(
        "select status from webhook_deliveries where endpoint_id = $1", endpoint_id
    )
    assert row["status"] == "cancelled"


async def test_e2e_endpoint_url_edit_applies_to_queued_retries(pg, repo, dispatcher_factory):
    endpoint_id = await _register_endpoint(pg, url="https://hooks.example.com/old")
    batch_id, items = await _seed_batch(pg, n_items=1, webhook_mode="svix")
    await _finish_all(repo, batch_id, items)
    await pg.execute(
        "update webhook_endpoints set url = 'https://hooks.example.com/new' where id = $1",
        endpoint_id,
    )
    receiver = _Receiver()
    dispatcher = dispatcher_factory(receiver)
    try:
        await _drain(dispatcher)
    finally:
        await dispatcher.stop()
    assert receiver.requests[0].headers["host"] == "hooks.example.com"
    assert receiver.requests[0].url.path == "/new"
