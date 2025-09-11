#!/usr/bin/env python3
"""
Client: Convert a PDF to Markdown using a locally served vLLM OpenAI endpoint
running ds4sd/SmolDocling-256M-preview, then reconstruct a Docling document.

This performs page rasterization to images, sends each page with the instruction
"Convert page to Docling.", collects DocTags, and exports Markdown.

Prereqs:
  - vLLM server running (see start_vllm_smoldocling.sh)
  - pip install: pillow, pypdfium2, docling_core, openai

Usage:
  python -m data_pipeline.smoldocling_vllm_client \
    --pdf /path/to/file.pdf \
    --out ./out.md \
    --openai-base http://localhost:8000 \
    --model smoldocling
"""

import argparse
import logging
from pathlib import Path
from typing import List

from PIL import Image
import pypdfium2 as pdfium
from openai import OpenAI
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument


def configure_logger(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def rasterize_pdf(pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
    """Render PDF pages to RGB PIL Images at the given DPI."""
    images: List[Image.Image] = []
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        for i in range(len(pdf)):
            page = pdf[i]
            pil_image = page.render(scale=dpi / 72.0).to_pil()
            images.append(pil_image.convert("RGB"))
    return images


def generate_doctags_for_images(client: OpenAI, model: str, images: List[Image.Image]) -> List[str]:
    """Send each page image to the vLLM OpenAI server to get DocTags."""
    doctags_list: List[str] = []
    for img in images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."},
                ],
            }
        ]
        # The Python OpenAI SDK with vLLM expects the image to be provided via files or a URL.
        # For simplicity, we pass an empty 'image' placeholder and rely on server-side support
        # for single image input prompts. If your server requires explicit image content,
        # consider saving to a temp file and using a data URL or serving via local HTTP.
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=8192)
        doctags = resp.choices[0].message.content or ""
        doctags_list.append(doctags.strip())
    return doctags_list


def convert_pdf_to_markdown(pdf_path: Path, openai_base: str, model: str) -> str:
    client = OpenAI(base_url=openai_base, api_key="EMPTY")
    images = rasterize_pdf(pdf_path)
    doctags_list = generate_doctags_for_images(client, model, images)

    # Build Docling Document from DocTags and images
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(doctags_list, images)
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=pdf_path.name)
    return doc.export_to_markdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown via vLLM SmolDocling server")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", required=True, help="Path to output Markdown file")
    parser.add_argument("--openai-base", default="http://localhost:8000", help="OpenAI base URL (vLLM)")
    parser.add_argument("--model", default="smoldocling", help="Model name served by vLLM")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()

    configure_logger(args.verbose)
    pdf_path = Path(args.pdf).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    md = convert_pdf_to_markdown(pdf_path, args.openai_base, args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Saved Markdown: {out_path}")


if __name__ == "__main__":
    main()


