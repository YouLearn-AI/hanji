-- Batch completion webhooks (split outbox).
--
-- Registered signed endpoints (Svix-wire-compatible delivery targets).
-- Managed by the `python -m extract.webhooks` CLI. Soft-deleted
-- (deleted_at) so queued deliveries keep a valid signing key;
-- disable = immediate kill switch.
CREATE TABLE IF NOT EXISTS "webhook_endpoints" (
	"id" text PRIMARY KEY NOT NULL,
	"customer_id" text NOT NULL,
	"url" text NOT NULL,
	"secret_ciphertext" text NOT NULL,
	"secret_key_id" text NOT NULL,
	"old_secret_ciphertext" text,
	"old_secret_expires_at" timestamp,
	"description" text,
	"enabled" boolean DEFAULT true NOT NULL,
	"deleted_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);

CREATE INDEX IF NOT EXISTS "webhook_endpoints_customer_idx" ON "webhook_endpoints" ("customer_id");

-- One logical event per batch terminal transition. `id` IS the svix-id
-- message id (stable across retries; the consumer's idempotency key).
-- `body` is the immutable UTF-8 JSON the signature covers — every attempt
-- sends these bytes verbatim; never re-serialize.
-- UNIQUE(batch_id, event_type) makes the enqueue exactly-once even under
-- concurrent finishers and the reconciler.
-- batch_id is NULL for synthetic events (webhook.ping test sends); NULLs are
-- distinct under the unique index so pings never collide with each other.
CREATE TABLE IF NOT EXISTS "webhook_events" (
	"id" text PRIMARY KEY NOT NULL,
	"customer_id" text NOT NULL,
	"batch_id" text,
	"event_type" text NOT NULL,
	"body" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);

ALTER TABLE "webhook_events" ADD CONSTRAINT "webhook_events_batch_id_extract_batches_id_fk" FOREIGN KEY ("batch_id") REFERENCES "extract_batches"("id") ON DELETE cascade ON UPDATE no action;

CREATE UNIQUE INDEX IF NOT EXISTS "webhook_events_batch_event_idx" ON "webhook_events" ("batch_id","event_type");

-- One row per destination. endpoint_id set for svix/registered deliveries,
-- NULL for direct (the destination is the batch's inline webhook_url,
-- snapshotted into `url` at enqueue). Claimed by the dispatcher with
-- FOR UPDATE SKIP LOCKED (pending due OR delivering with expired lease).
CREATE TABLE IF NOT EXISTS "webhook_deliveries" (
	"id" text PRIMARY KEY NOT NULL,
	"event_id" text NOT NULL,
	"endpoint_id" text,
	"customer_id" text NOT NULL,
	"url" text NOT NULL,
	"signed" boolean DEFAULT true NOT NULL,
	"status" text DEFAULT 'pending' NOT NULL,
	"attempts" integer DEFAULT 0 NOT NULL,
	"next_attempt_at" timestamp DEFAULT now() NOT NULL,
	"lease_token" text,
	"lease_expires_at" timestamp,
	"last_status_code" integer,
	"last_error" text,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"succeeded_at" timestamp
);

ALTER TABLE "webhook_deliveries" ADD CONSTRAINT "webhook_deliveries_event_id_webhook_events_id_fk" FOREIGN KEY ("event_id") REFERENCES "webhook_events"("id") ON DELETE cascade ON UPDATE no action;
ALTER TABLE "webhook_deliveries" ADD CONSTRAINT "webhook_deliveries_endpoint_id_webhook_endpoints_id_fk" FOREIGN KEY ("endpoint_id") REFERENCES "webhook_endpoints"("id") ON DELETE no action ON UPDATE no action;

CREATE UNIQUE INDEX IF NOT EXISTS "webhook_deliveries_event_endpoint_idx" ON "webhook_deliveries" ("event_id","endpoint_id");
-- SQL NULLs are distinct, so the composite unique above doesn't cover direct
-- rows; this partial index keeps direct single-destination per event.
CREATE UNIQUE INDEX IF NOT EXISTS "webhook_deliveries_event_direct_idx" ON "webhook_deliveries" ("event_id") WHERE "endpoint_id" IS NULL;
CREATE INDEX IF NOT EXISTS "webhook_deliveries_claim_idx" ON "webhook_deliveries" ("status","next_attempt_at");
CREATE INDEX IF NOT EXISTS "webhook_deliveries_customer_created_idx" ON "webhook_deliveries" ("customer_id","created_at");
