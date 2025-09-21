import os
import json
import base64
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from logs import log
from dotenv import load_dotenv 
from docling.document_converter import DocumentConverter
from minio import Minio
from PIL import Image  # Pillow already in requirements
import io as _io
import re
from tempfile import NamedTemporaryFile

load_dotenv()  # load variables from .env if present
DEFAULT_BUCKET = os.getenv("MINIO_BUCKET")




PDF_PATTERN = re.compile(r"(^|.*?/)([^/]+)/pdf/\2\.pdf$")  # matches any .../<paperId>/pdf/<paperId>.pdf capturing optional leading path


def list_paper_pdf_jobs(client: Minio, bucket: str, prefix: str = "") -> list[dict]:
    jobs: list[dict] = []
    seen_pdf_paths: set[str] = set()
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        name = obj.object_name
        if not name.endswith('.pdf') or '/pdf/' not in name:
            continue
        parts = name.split('/')
        if len(parts) < 3:
            continue
        if parts[-2] != 'pdf':
            continue
        filename = parts[-1]
        if not filename.endswith('.pdf'):
            continue
        paper_id_candidate = filename[:-4]
        if parts[-3] != paper_id_candidate:
            # Not matching pattern
            continue
        base_path = '/'.join(parts[:-2])  # includes paper_id folder
        pdf_object = name
        if pdf_object in seen_pdf_paths:
            continue
        seen_pdf_paths.add(pdf_object)
        jobs.append({
            'paper_id': paper_id_candidate,
            'base_path': base_path,
            'pdf_object': pdf_object,
        })
    # Stable order: sort by base_path then paper_id
    jobs.sort(key=lambda j: (j['base_path'], j['paper_id']))
    return jobs


def build_output_path(output_dir: Path, pdf_path: Path) -> Path:
    """Build output .md path mirroring the input filename (without extension)."""
    stem = pdf_path.stem
    return output_dir / f"{stem}.md"


def object_exists(client: Minio, bucket: str, object_name: str) -> bool:
    try:
        client.stat_object(bucket, object_name)
        return True
    except Exception:  # noqa: BLE001
        return False


def build_minio_client() -> Minio:
    return Minio(**{"endpoint": str(os.getenv("MINIO_ENDPOINT")),
                    "access_key": str(os.getenv("MINIO_ACCESS_KEY")),
                    "secret_key": str(os.getenv("MINIO_SECRET_KEY")),
                    "secure": False
                    })

def ensure_bucket(client: Minio, bucket: str):
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed ensuring bucket {bucket}: {e}")
    
def _mime_from_ext(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "gif":
        return "image/gif"
    if ext == "bmp":
        return "image/bmp"
    if ext == "tiff":
        return "image/tiff"
    if ext == "webp":
        return "image/webp"
    return f"image/{ext}"

def upload_bytes(client: Minio, bucket: str, object_name: str, data: bytes, content_type: str):
    try:
        import io as _io
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=_io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Upload failed for {object_name}: {e}")


def image_filename(page: int, idx: int, raw_bytes: bytes, ext: str) -> str:
    sha16 = hashlib.sha256(raw_bytes).hexdigest()[:16]
    return f"fig_p{page}_{idx}_{sha16}.{ext}"


"""Docling PDF parsing and MinIO upload script (local file writes removed)."""


def run(
    bucket: str = DEFAULT_BUCKET,
    prefix: str = "",
    overwrite: bool = False,
    limit: Optional[int] = None,
    max_image_bytes: int = 15 * 1024 * 1024,
    embed_images: bool = False,
) -> dict:
    log.info("[DOCLING] Starting remote Docling PDF parsing pipeline…")
    if not bucket:
        raise RuntimeError("MINIO_BUCKET is not set (None). Set it in your environment or .env file as MINIO_BUCKET=<bucket_name>.")
    log.info(f"[DOCLING] Bucket: {bucket} prefix: '{prefix}'")

    minio_client = build_minio_client()
    ensure_bucket(minio_client, bucket)
    jobs = list_paper_pdf_jobs(minio_client, bucket, prefix=prefix)
    if limit is not None:
        jobs = jobs[: max(0, int(limit))]
    log.info(f"[DOCLING] Discovered {len(jobs)} PDF object(s) matching pattern")
    if not jobs:
        log.warning("[DOCLING] No PDFs found. Check: 1) MINIO_BUCKET value 2) DOCLING_PREFIX (often should be empty or 'papers/') 3) Object layout ends with /<paperId>/pdf/<paperId>.pdf")

    converter = DocumentConverter()

    converted = 0
    skipped = 0
    failed = 0

    for job in jobs:
        paper_id = job['paper_id']
        base_path = job['base_path']
        pdf_object = job['pdf_object']
        md_object = f"{base_path}/markdown/{paper_id}.md"
        images_prefix = f"{base_path}/images/"

        if not overwrite and object_exists(minio_client, bucket, md_object):
            skipped += 1
            log.debug(f"[DOCLING] Skip {paper_id}: markdown already exists at {md_object}")
            continue
        # Download PDF
        try:
            resp = minio_client.get_object(bucket, pdf_object)
            pdf_bytes = resp.read()
            resp.close()
            resp.release_conn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.warning(f"[DOCLING] Failed download {pdf_object}: {e}")
            continue
        # Write to temp file for Docling
        try:
            with NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
                tmp.write(pdf_bytes)
                tmp.flush()
                # Reuse the same converter instead of instantiating per file
                conversion = converter.convert(tmp.name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.warning(f"[DOCLING] Conversion failed {paper_id}: {e}")
            continue
        # Export markdown
        try:
            markdown_text = conversion.document.export_to_markdown()
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.warning(f"[DOCLING] Markdown export failed {paper_id}: {e}")
            continue
        # Upload markdown
        try:
            upload_bytes(minio_client, bucket, md_object, markdown_text.encode('utf-8'), 'text/markdown')
        except Exception as e:  # noqa: BLE001
            log.warning(f"[DOCLING] Upload markdown failed {paper_id}: {e}")
        # Images extraction
        images_uploaded: List[str] = []
        images_detail: List[Dict[str, Any]] = []
        seen_hashes: set[str] = set()
        try:
            for p_idx, page in enumerate(getattr(conversion.document, 'pages', [])):
                if hasattr(page, 'images'):
                    candidates = page.images
                elif hasattr(page, 'elements'):
                    candidates = [el for el in page.elements if getattr(el, 'image', None)]
                else:
                    candidates = []
                for i_idx, img in enumerate(candidates):
                    raw = None
                    if hasattr(img, 'raw_bytes') and img.raw_bytes:
                        raw = img.raw_bytes
                    elif hasattr(img, 'bytes') and img.bytes:
                        raw = img.bytes
                    elif hasattr(img, 'image') and getattr(img.image, 'data', None):
                        try:
                            raw = base64.b64decode(img.image.data)
                        except Exception:
                            raw = None
                    if not raw:
                        continue
                    if max_image_bytes and len(raw) > max_image_bytes:
                        log.debug(f"[DOCLING] Skip oversize image {len(raw)} bytes {paper_id} p{p_idx} i{i_idx}")
                        continue
                    sha256_full = hashlib.sha256(raw).hexdigest()
                    if sha256_full in seen_hashes:
                        continue
                    seen_hashes.add(sha256_full)
                    ext = 'png'
                    fmt = None
                    width = height = None
                    mode = None
                    try:
                        with Image.open(_io.BytesIO(raw)) as im:
                            fmt = (im.format or '').lower()
                            if fmt == 'jpeg':
                                ext = 'jpg'
                            elif fmt in {'png','gif','bmp','tiff','webp','jpg','jpeg'}:
                                ext = 'jpg' if fmt == 'jpeg' else fmt
                            width, height = im.size
                            mode = im.mode
                    except Exception:
                        pass
                    fname = image_filename(p_idx, i_idx, raw, ext)
                    try:
                        upload_bytes(minio_client, bucket, f"{images_prefix}{fname}", raw, _mime_from_ext(ext))
                        images_uploaded.append(fname)
                        images_detail.append({
                            'file': fname,
                            'page': p_idx,
                            'index': i_idx,
                            'bytes': len(raw),
                            'hash': sha256_full[:32],
                            'format': fmt,
                            'width': width,
                            'height': height,
                            'mode': mode,
                        })
                    except Exception as ie:  # noqa: BLE001
                        log.debug(f"[DOCLING] Skip image upload {paper_id} p{p_idx} i{i_idx}: {ie}")
        except Exception:
            pass
        # Optional embed
        if embed_images and images_uploaded:
            append_lines = ["\n\n## Extracted Figures\n"]
            for det in images_detail:
                append_lines.append(f"![fig p{det['page']} i{det['index']}](/images/{det['file']})")
            markdown_text_final = markdown_text + "\n" + "\n".join(append_lines) + "\n"
            try:
                upload_bytes(minio_client, bucket, md_object, markdown_text_final.encode('utf-8'), 'text/markdown')
            except Exception as e:  # noqa: BLE001
                log.debug(f"[DOCLING] Re-upload markdown with embeds failed {paper_id}: {e}")
        converted += 1
        log.debug(f"[DOCLING] Processed {paper_id} (images={len(images_uploaded)})")

    stats = {
        "bucket": bucket,
        "prefix": prefix,
        "total": len(jobs),
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
    }

    log.info(
        f"[DOCLING] Completed. total={stats['total']} converted={converted} skipped={skipped} failed={failed}"
    )
    return stats

def main() -> None:
    bucket = os.getenv("MINIO_BUCKET", DEFAULT_BUCKET)
    prefix = os.getenv("DOCLING_PREFIX", "papers")
    overwrite = os.getenv("DOCLING_OVERWRITE", "false").lower() in {"1", "true", "yes"}
    limit_env = os.getenv("DOCLING_LIMIT")
    limit = int(limit_env) if limit_env and limit_env.isdigit() else None
    max_image_bytes = int(os.getenv("DOCLING_MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
    embed_images = os.getenv("DOCLING_EMBED_IMAGES", "false").lower() in {"1", "true", "yes"}

    run(
        bucket=bucket,
        prefix=prefix,
        overwrite=overwrite,
        limit=limit,
        max_image_bytes=max_image_bytes,
        embed_images=embed_images,
    )

if __name__ == "__main__":
    main()