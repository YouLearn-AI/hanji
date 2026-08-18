"""Document converters that emit PDF bytes.

Each converter takes an input (URL, path, or bytes) and returns PDF bytes
that the main PDF extractor can operate on.

  - ``pptx``  — LibreOffice headless
  - ``docx``  — LibreOffice headless
  - ``image`` — PNG / JPEG / WebP / TIFF / HEIC-HEIF / BMP → PDF via
    Pillow + PyMuPDF, one page per frame (multi-frame TIFF → multi-page PDF)
"""
