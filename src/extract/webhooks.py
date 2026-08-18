"""Manage registered webhook endpoints from the command line.

The hosted service managed signed webhook endpoints from a dashboard; this
build manages them here, against the same tables the dispatcher reads::

    python -m extract.webhooks add https://example.com/hooks [--description ...]
    python -m extract.webhooks list
    python -m extract.webhooks disable <endpoint_id>
    python -m extract.webhooks enable <endpoint_id>
    python -m extract.webhooks delete <endpoint_id>
    python -m extract.webhooks rotate <endpoint_id>
    python -m extract.webhooks ping <endpoint_id>
    python -m extract.webhooks deliveries [--limit N]

``add`` and ``rotate`` print the ``whsec_…`` signing secret ONCE — store it
in the consumer's secret manager; deliveries are signed with it
(Svix-wire-compatible, verify with the ``svix`` library).

Batches opt in per request: ``webhook: {"mode": "svix"}`` on
``POST /v1/batches`` delivers to every enabled endpoint registered here.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from extract.clients.webhook_secrets import from_settings as crypto_from_settings
from extract.config import settings
from extract.core.webhooks import (
    generate_secret,
    new_endpoint_id,
    validate_webhook_url,
)
from extract.repos.webhooks import WebhookRepo
from extract.repos.webhooks import from_settings as repo_from_settings

#: Must match the constant the API stamps on batch rows
#: (`extract.api.routes._helpers.DEFAULT_CUSTOMER_ID`).
from extract.api.routes._helpers import DEFAULT_CUSTOMER_ID


async def _add(repo: WebhookRepo, url: str, description: str | None) -> int:
    if not settings.EXTRACT_WEBHOOK_ALLOW_PRIVATE_URLS:
        error = validate_webhook_url(url)
        if error:
            print(f"refused: {error}", file=sys.stderr)
            print(
                "(EXTRACT_WEBHOOK_ALLOW_PRIVATE_URLS=true lifts this for "
                "private-network destinations)",
                file=sys.stderr,
            )
            return 1
    count = await repo.count_endpoints(customer_id=DEFAULT_CUSTOMER_ID)
    if count >= settings.EXTRACT_WEBHOOK_MAX_ENDPOINTS:
        print(
            f"refused: already {count} endpoints "
            f"(EXTRACT_WEBHOOK_MAX_ENDPOINTS={settings.EXTRACT_WEBHOOK_MAX_ENDPOINTS})",
            file=sys.stderr,
        )
        return 1
    crypto = crypto_from_settings()
    secret = generate_secret()
    ciphertext, key_id = await crypto.encrypt(secret)
    row = await repo.create_endpoint(
        endpoint_id=new_endpoint_id(),
        customer_id=DEFAULT_CUSTOMER_ID,
        url=url,
        description=description,
        secret_ciphertext=ciphertext,
        secret_key_id=key_id,
    )
    print(f"endpoint {row['id']} -> {row['url']}")
    print(f"signing secret (shown once, store it now): {secret}")
    return 0


async def _list(repo: WebhookRepo) -> int:
    rows = await repo.list_endpoints(customer_id=DEFAULT_CUSTOMER_ID)
    if not rows:
        print("no endpoints registered")
        return 0
    for r in rows:
        state = "enabled" if r["enabled"] else "DISABLED"
        desc = f"  # {r['description']}" if r["description"] else ""
        print(f"{r['id']}  {state}  {r['url']}{desc}")
    return 0


async def _set_enabled(repo: WebhookRepo, endpoint_id: str, enabled: bool) -> int:
    row = await repo.update_endpoint(
        endpoint_id=endpoint_id, customer_id=DEFAULT_CUSTOMER_ID, enabled=enabled
    )
    if row is None:
        print("endpoint not found", file=sys.stderr)
        return 1
    print(f"endpoint {row['id']} {'enabled' if row['enabled'] else 'disabled'}")
    return 0


async def _delete(repo: WebhookRepo, endpoint_id: str) -> int:
    ok = await repo.soft_delete_endpoint(
        endpoint_id=endpoint_id, customer_id=DEFAULT_CUSTOMER_ID
    )
    if not ok:
        print("endpoint not found", file=sys.stderr)
        return 1
    print(f"endpoint {endpoint_id} deleted (queued deliveries keep their signing key)")
    return 0


async def _rotate(repo: WebhookRepo, endpoint_id: str) -> int:
    crypto = crypto_from_settings()
    secret = generate_secret()
    ciphertext, key_id = await crypto.encrypt(secret)
    ok = await repo.rotate_endpoint_secret(
        endpoint_id=endpoint_id,
        customer_id=DEFAULT_CUSTOMER_ID,
        new_ciphertext=ciphertext,
        new_key_id=key_id,
    )
    if not ok:
        print("endpoint not found", file=sys.stderr)
        return 1
    print(f"rotated. new signing secret (shown once): {secret}")
    print("the previous secret keeps verifying for 24h (dual-signed deliveries)")
    return 0


async def _ping(repo: WebhookRepo, endpoint_id: str) -> int:
    event_id = await repo.create_ping(
        customer_id=DEFAULT_CUSTOMER_ID, endpoint_id=endpoint_id
    )
    if event_id is None:
        print("endpoint not found", file=sys.stderr)
        return 1
    print(f"queued webhook.ping event {event_id} (the worker's dispatcher delivers it)")
    return 0


async def _deliveries(repo: WebhookRepo, limit: int) -> int:
    rows = await repo.list_deliveries(customer_id=DEFAULT_CUSTOMER_ID, limit=limit)
    if not rows:
        print("no deliveries")
        return 0
    for r in rows:
        dest = r["endpoint_id"] or "direct"
        print(
            f"{r['created_at']:%Y-%m-%d %H:%M:%S}  {r['id']}  {r['event_type']:<12} "
            f"{r['status']:<10} attempts={r['attempts']} dest={dest} "
            f"last_status={r['last_status_code']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m extract.webhooks", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_add = sub.add_parser("add", help="register an endpoint; prints its signing secret once")
    p_add.add_argument("url")
    p_add.add_argument("--description")
    sub.add_parser("list", help="list registered endpoints")
    for name in ("disable", "enable", "delete", "rotate", "ping"):
        p = sub.add_parser(name)
        p.add_argument("endpoint_id")
    p_del = sub.add_parser("deliveries", help="recent delivery attempts")
    p_del.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    if not settings.DATABASE_URL:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 2
    repo = repo_from_settings()

    async def _run() -> int:
        try:
            if args.command == "add":
                return await _add(repo, args.url, args.description)
            if args.command == "list":
                return await _list(repo)
            if args.command == "disable":
                return await _set_enabled(repo, args.endpoint_id, False)
            if args.command == "enable":
                return await _set_enabled(repo, args.endpoint_id, True)
            if args.command == "delete":
                return await _delete(repo, args.endpoint_id)
            if args.command == "rotate":
                return await _rotate(repo, args.endpoint_id)
            if args.command == "ping":
                return await _ping(repo, args.endpoint_id)
            if args.command == "deliveries":
                return await _deliveries(repo, args.limit)
            raise AssertionError(args.command)
        finally:
            await repo.aclose()

    return asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())
