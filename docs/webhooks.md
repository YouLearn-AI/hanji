# Webhooks

Get a signed `batch.update` event POSTed to you the moment an async batch
reaches a terminal state, instead of polling. The signature scheme is
**wire-compatible with [Svix](https://www.svix.com/)** — if you already
verify Svix-style webhooks, your verification code works unchanged with the
standard open-source `svix` library.

## Setup

1. Register an endpoint (shown once: its `whsec_…` signing secret — store
   it as a secret in your app):

   ```bash
   python -m extract.webhooks add https://example.com/webhooks/extract
   # endpoint whe_…  ->  https://example.com/webhooks/extract
   # signing secret (shown once, store it now): whsec_…
   ```

   (`list`, `disable`, `enable`, `delete`, `rotate`, `ping`, and
   `deliveries` subcommands manage the rest. `rotate` dual-signs
   deliveries with old + new secrets for 24 hours.)

2. Opt any batch in with a `webhook` field:

   ```bash
   curl -X POST http://localhost:8080/v1/batches \
     -H "Content-Type: application/json" \
     -d '{
       "source": { "type": "files", "file_ids": ["file_abc"] },
       "metadata": { "job_id": "PA-1234" },
       "webhook": { "mode": "svix" }
     }'
   ```

Webhooks are **opt-in per batch** — a batch with no `webhook` field never
fires one, even with endpoints registered. When you do opt in, every
enabled endpoint receives the event.

**Direct mode** (prototyping): `"webhook": {"mode": "direct", "url":
"https://..."}` sends the same event, unsigned, to a per-batch inline URL —
no registration step. Authenticate it by round-tripping a token of your own
through `metadata`. Registered + signed is the production path.

## The event

```json
{
  "type": "batch.update",
  "batch_id": "batch_abc",
  "status": "partially_failed",
  "counts": { "pending": 0, "running": 0, "succeeded": 4, "failed": 1, "cancelled": 0 },
  "total_items": 5,
  "metadata": { "job_id": "PA-1234" },
  "completed_at": "2026-08-05T12:34:56Z",
  "results_expires_at": "2026-08-08T12:34:56Z"
}
```

The payload is deliberately thin: ids, status, counts, and your echoed
`metadata` — never results inline. To get the extraction, call
`GET /v1/batches/{batch_id}` and fetch each item's `result_url`.
`results_expires_at` is the deadline after which those results are gone.

> **`status` has four terminal values, not two.**
>
> | `status` | Meaning |
> | --- | --- |
> | `completed` | Every item succeeded. |
> | `partially_failed` | **Terminal and done** — some items succeeded and some did **not** (failed **or** were cancelled; `counts.failed` may be `0`). Results for the succeeded items are ready. |
> | `failed` | No item succeeded, no cancellations. |
> | `cancelled` | No item succeeded and at least one was cancelled. |
>
> If you're porting a handler that branches `if status == "Completed"`, it
> will silently ignore `partially_failed` batches and drop the results of
> every document that *did* succeed. Treat **both** `completed` and
> `partially_failed` as "results are ready — inspect `counts`."

## Handling it

Verify the signature, branch on `status`, return `2xx` fast, do the real
work after. Headers: `svix-id`, `svix-timestamp`, `svix-signature` (plus
the Standard-Webhooks aliases `webhook-id` / `webhook-timestamp` /
`webhook-signature`).

```python
import os
from flask import Flask, request, jsonify
from svix.webhooks import Webhook, WebhookVerificationError

app = Flask(__name__)
WEBHOOK_SECRET = os.environ["EXTRACT_WEBHOOK_SECRET"]  # whsec_… from `add`

@app.post("/webhooks/extract")
def handle():
    # Pass the RAW body and the untouched headers — signature verification
    # is sensitive to any change to the body bytes.
    try:
        payload = Webhook(WEBHOOK_SECRET).verify(request.get_data(), request.headers)
    except WebhookVerificationError:
        return jsonify(error="invalid signature"), 401
    if payload["type"] == "webhook.ping":   # health-check sends
        return jsonify(received=True), 200
    svix_id = request.headers["svix-id"]    # stable dedup key across retries
    if payload["status"] in ("completed", "partially_failed"):
        enqueue_fetch(payload["batch_id"], idempotency_key=svix_id)
    else:
        mark_terminal(payload["batch_id"], payload["status"], idempotency_key=svix_id)
    return jsonify(received=True), 200
```

## Delivery semantics

- **At-least-once.** Dedup on `svix-id` — it is stable across every retry
  of one event.
- **Retries**: after a failed attempt (anything but a 2xx within 15
  seconds; redirects are failures), delivery retries on a ladder of
  5s, 5m, 30m, 2h, 5h, 10h, 10h — 8 attempts over ~27.5 hours — then the
  delivery is marked failed. `python -m extract.webhooks deliveries` shows
  the attempt history; a failed delivery can be requeued from there.
- **Disable is immediate**: `python -m extract.webhooks disable <id>` stops
  queued deliveries at claim time.
- **Destinations must be public HTTPS hostnames** (SSRF protection: the
  dispatcher re-resolves and pins a vetted public IP on every attempt).
  Self-hosting on a private network? `EXTRACT_WEBHOOK_ALLOW_PRIVATE_URLS=true`
  lifts this — leave it off whenever endpoint URLs come from untrusted
  users.
- A reconciler sweeps for opted-in batches that went terminal without an
  event (crash windows) and enqueues the missing event, bounded to a
  24-hour lookback.
