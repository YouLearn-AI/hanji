.PHONY: setup dev test docker-build docker-run migrate worker

setup:            ## install deps (uv) — LibreOffice needed for DOCX/PPTX: brew install libreoffice
	uv sync --extra api --extra aws --extra batch

dev:              ## run the API locally on :8001
	uv run fastapi dev src/extract/api/app.py --port 8001

test:             ## offline test suite
	uv run pytest

docker-build:
	docker build -t extract-api .

docker-run:
	docker run --rm -p 8080:8080 --env-file .env extract-api

migrate:          ## apply batch-lane DB migrations (needs DATABASE_URL)
	uv run python -m extract.migrate

worker:           ## run the async batch worker (needs DATABASE_URL + buckets)
	uv run python -m extract.workers.batch_worker
