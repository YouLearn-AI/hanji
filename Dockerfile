FROM python:3.12-slim-bookworm

# uv installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# System deps:
#   - LibreOffice for PPTX / DOCX → PDF conversion
#   - Fonts (CJK, Arabic, Hebrew, Indic, European) for robust conversion
#   - libheif for HEIC decoding (pillow-heif links against it)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libreoffice-core libreoffice-common libreoffice-writer libreoffice-impress libreoffice-calc \
        libxinerama1 libxrandr2 libxcomposite1 libglu1-mesa libsm6 \
        libheif1 \
        fonts-dejavu fonts-liberation fonts-liberation2 fonts-dejavu-extra \
        fonts-noto-cjk fonts-noto fonts-nanum \
        fonts-kacst fonts-sil-abyssinica fonts-thai-tlwg \
        locales fontconfig \
 && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

RUN useradd --create-home appuser
WORKDIR /home/appuser

ENV UV_HTTP_TIMEOUT=3600 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --chown=appuser:appuser pyproject.toml uv.lock* ./
USER appuser
# Dependency layer only (the project itself needs the full source, which
# isn't copied yet — install it after the COPY below).
RUN uv venv /home/appuser/.venv \
 && uv sync --frozen --no-dev --no-install-project --extra api --extra aws --extra batch || \
    uv sync --no-dev --no-install-project --extra api --extra aws --extra batch

ENV PATH="/home/appuser/.venv/bin:$PATH"
ENV PORT=8080

COPY --chown=appuser:appuser . .

RUN uv sync --frozen --no-dev --extra api --extra aws --extra batch || \
    uv sync --no-dev --extra api --extra aws --extra batch

EXPOSE 8080

CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "src/extract/api/app.py"]
