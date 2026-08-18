"""Shared LibreOffice subprocess driver for PPTX / DOCX."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

from extract.core.errors import ExtractionFailed
from extract.core.io import load_bytes
from extract.logger import get_logger

logger = get_logger()


async def convert_office_to_pdf(
    *,
    url: str | None = None,
    path: str | None = None,
    data: bytes | None = None,
    max_size: int | None,
    input_ext: str,
    block_private: bool = True,
) -> bytes:
    input_bytes = await load_bytes(
        url=url, path=path, data=data, max_size=max_size, block_private=block_private
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / f"input.{input_ext}"
        input_path.write_bytes(input_bytes)

        with tempfile.TemporaryDirectory() as profile_dir:
            cmd = [
                "soffice",
                "--headless",
                f"-env:UserInstallation=file://{profile_dir}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(tmp_dir),
                str(input_path),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise ExtractionFailed(
                    f"LibreOffice conversion failed: {stderr.decode(errors='replace')}"
                )

        pdf_path = input_path.with_suffix(".pdf")
        if not pdf_path.exists():
            raise ExtractionFailed(
                f"LibreOffice did not produce a PDF at {pdf_path}"
            )
        return pdf_path.read_bytes()
