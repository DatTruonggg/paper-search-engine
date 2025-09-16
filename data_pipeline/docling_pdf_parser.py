import argparse
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
from logs import log
from docling.document_converter import DocumentConverter

def export_pdf_to_markdown(converter: DocumentConverter, pdf_path: Path, markdown_path: Path) -> dict:
    """
    Convert a single PDF to markdown using Docling and write to disk.

    Returns a dict with status information for logging and metrics.
    """
    result = {
        "pdf": str(pdf_path),
        "markdown": str(markdown_path),
        "status": "pending",
        "bytes": 0,
    }

    try:
        conversion = converter.convert(str(pdf_path))
        markdown_text = conversion.document.export_to_markdown()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_text, encoding='utf-8')
        result["bytes"] = markdown_path.stat().st_size
        result["status"] = "ok"
        return result
    except Exception as exc:  # Only logic-related comment: capture and report conversion errors
        result["status"] = "error"
        result["error"] = str(exc)
        return result


def discover_pdfs(input_dir: Path, pattern: str) -> list[Path]:
    """Discover PDF files in the input directory matching a glob pattern."""
    return sorted(input_dir.glob(pattern))


def build_output_path(output_dir: Path, pdf_path: Path) -> Path:
    """Build output .md path mirroring the input filename (without extension)."""
    stem = pdf_path.stem
    return output_dir / f"{stem}.md"


def save_run_manifest(output_dir: Path, stats: dict) -> None:
    """Persist a small JSON manifest describing the run for auditability."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": stats.get("input_dir"),
        "output_dir": str(output_dir),
        "total": stats.get("total", 0),
        "converted": stats.get("converted", 0),
        "skipped": stats.get("skipped", 0),
        "failed": stats.get("failed", 0),
    }
    path = output_dir / f"docling_parse_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def run(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.pdf",
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """
    Perform batch conversion of PDFs under input_dir to markdown files under output_dir.

    Returns run statistics for logging and testing.
    """
    log.info("Starting Docling PDF parsing pipeline…")
    log.info(f"Input: {input_dir}")
    log.info(f"Output: {output_dir}")

    pdf_paths = discover_pdfs(input_dir, pattern)
    if limit is not None:
        pdf_paths = pdf_paths[: max(0, int(limit))]

    log.info(f"Discovered {len(pdf_paths)} PDF(s)")
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()

    converted = 0
    skipped = 0
    failed = 0

    for pdf in pdf_paths:
        markdown_path = build_output_path(output_dir, pdf)

        if markdown_path.exists() and not overwrite:
            skipped += 1
            continue

        res = export_pdf_to_markdown(converter, pdf, markdown_path)
        if res["status"] == "ok":
            converted += 1
            log.debug(f"Converted: {pdf} -> {markdown_path} ({res['bytes']} bytes)")
        else:
            failed += 1
            log.warning(f"Failed: {pdf} -> {res.get('error', 'unknown error')}")

    stats = {
        "input_dir": str(input_dir),
        "total": len(pdf_paths),
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
    }

    log.info(
        f"Completed. total={stats['total']} converted={converted} skipped={skipped} failed={failed}"
    )
    save_run_manifest(output_dir, stats)
    return stats


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the parser script."""
    parser = argparse.ArgumentParser(description="Batch convert PDFs to markdown using Docling")
    parser.add_argument(
        "--input-dir",
        default="/Users/admin/code/cazoodle/data/pdfs",
        help="Directory containing input PDF files",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/processed/markdown",
        help="Directory to write .markdown files",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern to select PDFs under input-dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if they exist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of files to process",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    configure_log(args.verbose)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    run(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=args.pattern,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()


