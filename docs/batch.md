# Batch processing

Async batch extraction: upload many files (or hand over URLs), submit one
batch, poll a single endpoint — or get a [webhook](webhooks.md) — until it's
done. The per-item result JSON is exactly what `POST /v1/parse/file` would
have returned for that document.

The batch lane is optional. It needs three things the sync routes don't
(see [Running it](#running-it)): Postgres, an S3-compatible object store,
and the worker process.

## How it works

**By upload** — you control the bytes end to end:

```
1. POST /v1/files         (per file)   →  file_id + presigned upload URL
2. PUT <upload.url>       (per file)   →  upload the file bytes
3. POST /v1/batches       (once)       →  batch_id, status="pending"
4. GET  /v1/batches/{id}  (poll loop)  →  per-item status as items finish
5. GET  /v1/batches/{id}/items/{id}/result  →  302 redirect to the result JSON
```

**By URL** — skip steps 1-2 entirely; the worker fetches:

```
1. POST /v1/batches       (once)       →  {"source": {"type": "urls", "urls": [...]}}
2. GET  /v1/batches/{id}  (poll loop)
3. GET  /v1/batches/{id}/items/{id}/result
```

Properties worth knowing before you build:

- **Same input formats as sync.** PDF, PPTX, DOCX, and raster images
  (PNG, JPEG, WebP, TIFF, HEIC/HEIF, BMP).
- **Uploads go straight to the object store.** `POST /v1/files` returns a
  presigned URL and you PUT the bytes to it directly; they never transit
  the API.
- **No separate "confirm upload" step.** Hand the `file_id`s straight to
  `POST /v1/batches` once your PUTs return. (`GET /v1/files/{file_id}`
  reports upload `status` and a `sha256` if you want to check first.)
- **3-day retention window** on batches (`expires_at`). Configure a
  matching object-lifecycle rule on your buckets if you want inputs and
  result blobs cleaned up automatically
  (`EXTRACT_BATCH_DEFAULT_TTL_SECONDS` to change it).
- **Retries are safe.** Send an `Idempotency-Key` header with
  `POST /v1/batches`. Retrying with the same key within 3 days returns the
  same batch instead of creating a duplicate.
- **Limits**: 150 MB / 2,000 pages per file; up to 10,000 `file_ids` or
  100 `urls` per batch; `metadata` up to 16 KB serialized.

## End-to-end example

```python
import asyncio, time
from pathlib import Path

import httpx

API = "http://localhost:8080"


async def upload(client: httpx.AsyncClient, path: Path) -> str:
    meta = (await client.post(
        f"{API}/v1/files",
        json={"filename": path.name, "size_bytes": path.stat().st_size},
    )).json()
    async with httpx.AsyncClient() as raw:
        await raw.put(
            meta["upload"]["url"],
            content=path.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=600,
        )
    return meta["id"]


async def main(input_dir: str, output_dir: str) -> None:
    files = sorted(p for p in Path(input_dir).rglob("*.pdf") if p.is_file())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60) as client:
        sem = asyncio.Semaphore(10)
        async def _bound(p):
            async with sem: return await upload(client, p)
        file_ids = await asyncio.gather(*[_bound(p) for p in files])

        batch = (await client.post(
            f"{API}/v1/batches",
            headers={"Idempotency-Key": f"my-run-{int(time.time())}"},
            json={"source": {"type": "files", "file_ids": file_ids}},
        )).json()
        print("submitted", batch["id"], "with", batch["total_items"], "items")

        while True:
            state = (await client.get(f"{API}/v1/batches/{batch['id']}")).json()
            print(state["status"], state["counts"])
            if state["status"] in {"completed", "partially_failed", "failed",
                                   "cancelled", "expired"}:
                break
            await asyncio.sleep(3)

        for item in state["items"]:
            if item["status"] != "succeeded":
                continue
            r = await client.get(f"{API}{item['result_url']}", follow_redirects=True)
            (Path(output_dir) / f"{item['id']}.json").write_bytes(r.content)


asyncio.run(main("./inbox", "./results"))
```

## Submitting by URL

Pass `source: {"type": "urls", "urls": [...]}` and skip the upload step.
Each URL becomes one item; the worker fetches it when that item starts
processing, not when you submit.

- **Public and presigned URLs both work.** A presigned URL's signature
  lives in the query string; the worker uses the URL exactly as given.
  Give it enough TTL to survive queue time.
- **Source bytes are never written to storage.** The document is held only
  for the length of the extraction; only the *result* JSON is persisted.
- **Poll responses echo `url`** on each item with the query string
  stripped, so a presigned signature never shows up in a response
  (`items[].error.message` gets the same treatment).
- **Permanent fetch failures don't retry** — a bad URL, a 403/404, or an
  expired signature fails the item immediately as `url_fetch_failed`.
  Transient failures (5xx, timeouts) retry like any other item.

## Batch options

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | object | required | `{"type": "files", "file_ids": [...]}` or `{"type": "urls", "urls": [...]}` |
| `extract_text` | boolean | `true` | Set `false` to skip text spans in every item. |
| `extract_images` | boolean | `true` | Set `false` to skip figure extraction in every item. |
| `table_output_format` | `"markdown" \| "html"` | `"markdown"` | How table chunks are structured in each item's result. |
| `chunking` | `"none" \| "semantic"` | `"none"` | `"semantic"` adds RAG-ready `segments` to every item's result. |
| `chunk_size` | integer | `1000` | Target segment size in characters (±25% band); validated 200–8000 when chunking is enabled. |
| `metadata` | object | `null` | Arbitrary JSON echoed on every poll, in the batch list, and in webhook bodies. |
| `webhook` | object | `null` | Per-batch completion webhook opt-in — see [webhooks.md](webhooks.md). |

## Item lifecycle and errors

Items: `pending → running → succeeded | failed | cancelled` (a transient
failure returns the item to `pending` with a backoff; 3 attempts max).
Batches: `pending → running → completed | partially_failed | failed |
cancelled | expired`, derived from item counts; terminal states are sticky.

> **`partially_failed` is terminal and means "results are ready".** Some
> items succeeded and some did not — check `counts` for the breakdown. A
> handler that only branches on `completed` silently drops the successful
> results of a 4-of-5 batch.

`items[].error.code` values: `extraction_failed`, `unsupported_input`,
`document_too_large`, `page_limit_exceeded`, `ocr_provider_error`,
`upload_missing`, `url_fetch_failed`, `internal_error`
(`ocr_provider_error` and `internal_error` retry before going terminal).

## Polling cursor

`GET /v1/batches/{id}` returns items in `(updated_at, id)` order, paginated.
Pass back `next_cursor` as `?cursor=...` to fetch only what changed since
your last call; `limit` defaults to 100, caps at 500. Status polling is
rate-limited (200 req/s in-process; `EXTRACT_BATCH_POLL_RATE_LIMIT_PER_SECOND`)
— use [webhooks](webhooks.md) for completion notification in production.

`GET /v1/batches` lists batches newest first with the same cursor pattern,
an optional `?status=` filter, and a `limit` of up to 200 (default 50).

`POST /v1/batches/{id}/cancel` flips remaining `pending` items to
`cancelled`; items already `running` finish on their own.

## Running it

```bash
# 1. Postgres + MinIO + API + worker, all provisioned:
docker compose --profile batch up

# Or by hand:
export DATABASE_URL=postgres://...
export EXTRACT_UPLOADS_BUCKET=... EXTRACT_RESULTS_BUCKET=...
python -m extract.migrate                    # apply migrations
fastapi run src/extract/api/app.py           # the API
python -m extract.workers.batch_worker       # the worker (separate process)
```

Any S3-compatible store works: set `EXTRACT_S3_ENDPOINT_URL` for MinIO
etc., and `EXTRACT_S3_PUBLIC_ENDPOINT_URL` when clients reach the store on
a different host than the API/worker do (compose does this for you).
