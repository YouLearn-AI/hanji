# hanji

Document parsing and grounded schema extraction, built around
[**hanji-parse-4b**](https://huggingface.co/hanji-dev/hanji-parse-4b) — a
fine-tuned Qwen3-VL-4B vision-language model that reads document pages into
layout-grounded blocks.

Send a PDF, PPTX, DOCX, or image; get back every run of text, every table as
structured cells (plus a markdown rendering), and every figure, in reading
order, in one API call. Everything you get back is grounded: each chunk carries
its page number and bounding box, so you can always point back to the exact
spot on the page it came from.

This is the open-source release of the production pipeline that ran
[hanji.dev](https://hanji.dev) — the exact serving path, with the hosted
platform (auth, billing, telemetry) removed.

## The endpoints

| Route | What it does |
| --- | --- |
| `POST /v1/parse` | Parse a document by URL into chunks |
| `POST /v1/parse/file` | Parse an uploaded document (multipart) |
| `POST /v1/extract/schema` | Fill your JSON schema from a document URL, with citations |
| `POST /v1/extract/schema/file` | The multipart twin |
| `POST /v1/files` | Presigned upload slot for the async batch lane |
| `POST /v1/batches` (+ poll / result / cancel / list) | Async batch processing with completion [webhooks](docs/webhooks.md) |

The four sync routes need nothing but a model endpoint (and Gemini
credentials for schema extraction). The async batch lane is optional and
additionally needs Postgres + an S3-compatible object store + the worker
process — `docker compose --profile batch up` provisions all three; see
[docs/batch.md](docs/batch.md) and [docs/webhooks.md](docs/webhooks.md).

There is **no auth or billing** in this server. If you expose it publicly,
front it with your own gateway.

## Quickstart

You need two things:

1. **A parse-model endpoint** (`PARSE_MODEL_URL`) — an SGLang server running
   the [published weights](https://huggingface.co/hanji-dev/hanji-parse-4b).
   See [serving/SELF_HOSTING.md](serving/SELF_HOSTING.md); with a GPU and
   docker it is one command.
2. **A Gemini transport for schema extraction** (optional — parsing works
   without it): `GEMINI_API_KEY`, or a Vertex AI project via
   `GOOGLE_VERTEX_PROJECT`.

```bash
cp .env.example .env   # fill in PARSE_MODEL_URL (+ GEMINI_API_KEY for schema extraction)

# with docker (API + local GPU model server):
docker compose --profile gpu up

# or, API only, locally:
uv sync --extra api
uv run fastapi dev src/extract/api/app.py --port 8001
```

### Parse

```bash
curl -X POST http://localhost:8001/v1/parse/file -F "file=@paper.pdf"
```

```jsonc
{
  "chunks": [
    {
      "page_content": "Invoice Number: INV-2041\nIssue Date: 03/14/2026",
      "page_no": 1,
      "bbox": [72.0, 108.9, 353.2, 160.9],   // PDF points, [x0, y0, x1, y1]
      "chunk_type": "text",                  // text | table | image | key_value
      "confidence": 0.998
    },
    {
      "page_content": "| Item | Qty | Amount |\n| --- | --- | --- |\n| Widget A | 4 | $50.00 |",
      "page_no": 1,
      "bbox": [70.3, 208.1, 380.0, 310.5],
      "chunk_type": "table",
      "cells": [ /* per-cell rows/cols/spans */ ]
    }
  ]
}
```

Optional form/body fields: `table_output_format` (`markdown` | `html`),
`chunking` (`semantic` returns embed-ready `segments`), `include_content`
(whole document as one string), `extract_text` / `extract_images`.

### Schema extraction

Hand it your JSON schema; get back just those fields, each with a citation
verifying where on the page the value came from. Values whose citations cannot
be verified are nulled and listed in `ungrounded_fields` (set `strict=false` to
keep them), so a fabricated value never reaches you silently.

```bash
curl -X POST http://localhost:8001/v1/extract/schema/file \
  -F "file=@invoice.pdf" \
  -F 'schema_raw={
    "invoice_number": {"type": "string"},
    "total_due":      {"type": "string"},
    "line_items": {"type": "array", "items": {"type": "object", "properties": {
      "item": {"type": "string"}, "amount": {"type": "string"}}}}
  }'
```

```jsonc
{
  "values": {
    "invoice_number": "INV-2041",
    "total_due": "$118.00",
    "line_items": [ { "item": "Widget A", "amount": "$50.00" } ]
  },
  "evidence": {
    "invoice_number": [{ "page": 1, "bbox": [117.0, 140.0, 556.0, 199.0],
                         "text": "Invoice Number: INV-2041", "confidence": 0.9997 }]
  },
  "ungrounded_fields": [],
  "page_count": 2
}
```

No schema in mind? Send `auto_schema=true` and one is designed from the
document first, then filled the same grounded way (returned as
`generated_schema`).

## How it works

- **Parse**: every page is rasterized (≤2 MP) and read by the fine-tuned model
  into semantic blocks — `{bbox_2d, text_content}` in 0–1000 normalized
  coordinates, tables as GitHub-flavored markdown. A per-page Gemini fallback
  covers unusable reads; a numeric self-consistency gate re-reads
  numeric-dense pages; legal-document post-processing repairs transcripts and
  multi-panel layouts. Office documents convert through LibreOffice first.
- **Schema extraction**: the parse chunks (annotated with page positions and
  section tags) go to Gemini with your schema as a structured-output contract.
  Every returned value must quote its source chunk; quotes are verified
  against the document and unverifiable values are nulled. Citation boxes are
  tightened via Cloud Vision and low-confidence high-stakes fields are
  re-read from an upscaled crop — both optional refinements that fail open
  without their credentials.
- **Optional cross-read**: with AWS credentials (`uv sync --extra aws`),
  handwriting-heavy pages get a second, independent Textract read fused into
  the extraction context.

## Configuration

Everything is env-driven; see [.env.example](.env.example) for the variables
that matter and `src/extract/config.py` for the full list. Defaults are the
production-effective configuration the pipeline was measured and shipped with.

## Development

```bash
uv sync --extra api --extra aws --extra batch
uv run pytest            # offline suite; -m live for network tests
```

The batch lane's DB-backed tests need a throwaway Postgres:
`EXTRACT_TEST_DATABASE_URL=postgres://... uv run pytest` (they skip
otherwise).

## License

[Apache-2.0](LICENSE). The model weights at
[hanji-dev/hanji-parse-4b](https://huggingface.co/hanji-dev/hanji-parse-4b)
are Apache-2.0 as well (fine-tuned from
[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)).
