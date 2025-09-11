#!/usr/bin/env python3
"""
Test: Convert up to 10 PDFs to text using Docling and validate outputs.

This script selects at most 10 PDF files from the input directory, runs the
Docling-based conversion, and verifies that corresponding .md files exist in
the output directory. Exits with non-zero status on failure.

Usage example:
    python -m data_pipeline.test_docling_10_files \
      --input-dir "/Users/admin/code/cazoodle/data/pdfs" \
      --output-dir "./data/processed/md_test_10" \
      -vv
"""

import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

from data_pipeline.docling_pdf_parser import run as run_parse


def configure_logger(verbosity: int) -> None:
    """Configure logger level based on verbosity flag."""
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def discover_first_n_pdfs(input_dir: Path, n: int) -> list[Path]:
    """Return the first N PDFs (sorted) from the input directory."""
    return sorted(input_dir.glob("*.pdf"))[: max(0, int(n))]


def write_summary(output_dir: Path, summary: dict) -> None:
    """Persist JSON summary of the test run in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"docling_test_10_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    """Entry point to execute the 10-file conversion test."""
    parser = argparse.ArgumentParser(description="Test converting up to 10 PDFs to text using Docling")
    parser.add_argument(
        "--input-dir",
        default="/Users/admin/code/cazoodle/data/pdfs",
        help="Directory with PDF files",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/processed/md_test_10",
        help="Directory to write .md files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    args = parser.parse_args()

    configure_logger(args.verbose)
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    target_pdfs = discover_first_n_pdfs(input_dir, 10)
    if not target_pdfs:
        raise SystemExit("No PDF files found to test.")

    logger.info(f"Testing conversion on {len(target_pdfs)} PDF(s)")

    # Run conversion limited to the selected files only
    # Only logic-related comment: the parser's run() applies its own discovery and limit,
    # so we use a dedicated temp directory with just the selected files to simplify validation.
    temp_input_dir = output_dir / "_temp_input_selection"
    temp_input_dir.mkdir(parents=True, exist_ok=True)

    # Create hardlinks when possible to avoid copying large files
    created_links = []
    for pdf in target_pdfs:
        dst = temp_input_dir / pdf.name
        try:
            if dst.exists():
                continue
            dst.hardlink_to(pdf)
            created_links.append(dst)
        except Exception:
            # Fallback to copy if hardlink fails
            import shutil
            shutil.copy2(pdf, dst)
            created_links.append(dst)

    stats = run_parse(
        input_dir=temp_input_dir,
        output_dir=output_dir,
        pattern="*.pdf",
        overwrite=True,
        limit=10,
    )

    expected_md = [(output_dir / f"{p.stem}.md") for p in target_pdfs]
    missing = [p for p in expected_md if not p.exists() or p.stat().st_size == 0]

    summary = {
        "tested": len(target_pdfs),
        "converted": stats.get("converted", 0),
        "skipped": stats.get("skipped", 0),
        "failed": stats.get("failed", 0),
        "missing_outputs": [str(p) for p in missing],
    }
    write_summary(output_dir, summary)

    if missing or stats.get("failed", 0) > 0:
        logger.error(f"Test failed. Missing outputs: {len(missing)}, failed conversions: {stats.get('failed', 0)}")
        raise SystemExit(1)

    logger.info("Test passed: All expected outputs exist and are non-empty.")


if __name__ == "__main__":
    main()


