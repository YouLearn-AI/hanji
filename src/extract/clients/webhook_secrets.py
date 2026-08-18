"""Signing-secret storage codec for webhook endpoints.

The hosted service encrypted signing secrets at rest with KMS; this
self-hosted build stores them base64-obfuscated in the deployment's own
Postgres instead. That is deliberate: the secret only authenticates webhook
deliveries to endpoints the deployer registered themselves, and anyone with
read access to this database can already read every batch and result row.
The interface (``encrypt``/``decrypt`` + a recorded ``key_id``) is kept so a
real KMS/Fernet codec can slot in without touching callers or the schema.
"""

from __future__ import annotations

import base64

LOCAL_KEY_ID = "local"


class WebhookSecretCrypto:
    def __init__(self, *, key_id: str = LOCAL_KEY_ID) -> None:
        self._key_id = key_id

    @property
    def configured(self) -> bool:
        return True

    @property
    def key_id(self) -> str:
        return self._key_id

    async def encrypt(self, plaintext: str) -> tuple[str, str]:
        """Returns (ciphertext_b64, key_id)."""
        return base64.b64encode(plaintext.encode()).decode("ascii"), self._key_id

    async def decrypt(self, ciphertext_b64: str, *, key_id: str) -> str:  # noqa: ARG002
        return base64.b64decode(ciphertext_b64).decode()


def from_settings() -> WebhookSecretCrypto:
    return WebhookSecretCrypto()
