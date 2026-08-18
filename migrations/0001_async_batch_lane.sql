-- Async batch lane: uploaded files, batches, batch items.
--
-- Single-tenant translation of the hosted service's schema. `customer_id`
-- is kept as a column (every query scopes on it) but there is no customers
-- table — the API stamps a constant. Timestamps are TIMESTAMP WITHOUT TIME
-- ZONE storing UTC by convention; the Python side strips tzinfo at the
-- boundary (asyncpg refuses tz-aware datetimes on tz-naive columns).

CREATE TABLE IF NOT EXISTS "extract_files" (
	"id" text PRIMARY KEY NOT NULL,
	"customer_id" text NOT NULL,
	"phi_safe" boolean DEFAULT false NOT NULL,
	"s3_bucket" text NOT NULL,
	"s3_key" text NOT NULL,
	"filename" text,
	"content_type" text,
	"size_bytes" bigint NOT NULL,
	"sha256" text,
	"status" text DEFAULT 'pending_upload' NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"expires_at" timestamp NOT NULL
);

CREATE TABLE IF NOT EXISTS "extract_batches" (
	"id" text PRIMARY KEY NOT NULL,
	"customer_id" text NOT NULL,
	"phi_safe" boolean DEFAULT false NOT NULL,
	"status" text DEFAULT 'pending' NOT NULL,
	"idempotency_key" text,
	"total_items" integer NOT NULL,
	"counts_json" jsonb NOT NULL,
	"metadata_json" jsonb,
	"engine" text,
	"extract_text" boolean DEFAULT true NOT NULL,
	"extract_images" boolean DEFAULT true NOT NULL,
	"ocr" text DEFAULT 'auto' NOT NULL,
	"table_output_format" text DEFAULT 'markdown' NOT NULL,
	"chunking" text DEFAULT 'none' NOT NULL,
	"chunk_size" integer DEFAULT 1000 NOT NULL,
	"webhook_mode" text,
	"webhook_url" text,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"started_at" timestamp,
	"completed_at" timestamp,
	"expires_at" timestamp NOT NULL
);

CREATE TABLE IF NOT EXISTS "extract_batch_items" (
	"id" text PRIMARY KEY NOT NULL,
	"batch_id" text NOT NULL,
	"customer_id" text NOT NULL,
	"file_id" text,
	"url" text,
	"position" integer NOT NULL,
	"status" text DEFAULT 'pending' NOT NULL,
	"page_count" integer,
	"error_code" text,
	"error_message" text,
	"result_s3_bucket" text,
	"result_s3_key" text,
	"lease_token" text,
	"lease_expires_at" timestamp,
	"run_after" timestamp,
	"attempts" integer DEFAULT 0 NOT NULL,
	"started_at" timestamp,
	"completed_at" timestamp,
	"updated_at" timestamp DEFAULT now() NOT NULL
);

ALTER TABLE "extract_batch_items" ADD CONSTRAINT "extract_batch_items_batch_id_extract_batches_id_fk" FOREIGN KEY ("batch_id") REFERENCES "extract_batches"("id") ON DELETE cascade ON UPDATE no action;
ALTER TABLE "extract_batch_items" ADD CONSTRAINT "extract_batch_items_file_id_extract_files_id_fk" FOREIGN KEY ("file_id") REFERENCES "extract_files"("id") ON DELETE no action ON UPDATE no action;

CREATE INDEX IF NOT EXISTS "extract_files_customer_created_idx" ON "extract_files" ("customer_id","created_at");
CREATE INDEX IF NOT EXISTS "extract_batches_customer_created_idx" ON "extract_batches" ("customer_id","created_at");
CREATE UNIQUE INDEX IF NOT EXISTS "extract_batches_customer_idem_idx" ON "extract_batches" ("customer_id","idempotency_key");
CREATE UNIQUE INDEX IF NOT EXISTS "extract_batch_items_batch_position_idx" ON "extract_batch_items" ("batch_id","position");
CREATE INDEX IF NOT EXISTS "extract_batch_items_claim_idx" ON "extract_batch_items" ("status","run_after","id");
CREATE INDEX IF NOT EXISTS "extract_batch_items_customer_status_idx" ON "extract_batch_items" ("customer_id","status");
CREATE INDEX IF NOT EXISTS "extract_batch_items_batch_updated_idx" ON "extract_batch_items" ("batch_id","updated_at","id");
CREATE INDEX IF NOT EXISTS "extract_batch_items_lease_idx" ON "extract_batch_items" ("lease_expires_at");
