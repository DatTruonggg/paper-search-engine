#!/usr/bin/env python3
"""
Test runner script for paper search engine.
Provides convenient test execution with different options.
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"\n🔄 {description}")
        print("=" * 50)

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")

    required_packages = [
        "pytest",
        "elasticsearch",
        "transformers",
        "numpy",
        "torch"
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt -r tests/requirements.txt")
        return False

    print("✅ All dependencies available")
    return True


def check_services(host="localhost", es_port="9202", minio_port="9002"):
    """Check if required services are running"""
    print("🔍 Checking services...")

    import requests

    # Check Elasticsearch
    try:
        response = requests.get(f"http://{host}:{es_port}/_cluster/health", timeout=5)
        if response.status_code == 200:
            print("✅ Elasticsearch is running")
            es_ok = True
        else:
            print("❌ Elasticsearch is not healthy")
            es_ok = False
    except requests.RequestException:
        print("❌ Elasticsearch is not accessible")
        es_ok = False

    # Check MinIO
    try:
        response = requests.get(f"http://{host}:{minio_port}/minio/health/live", timeout=5)
        if response.status_code == 200:
            print("✅ MinIO is running")
            minio_ok = True
        else:
            print("❌ MinIO is not healthy")
            minio_ok = False
    except requests.RequestException:
        print("❌ MinIO is not accessible")
        minio_ok = False

    return es_ok, minio_ok


def run_unit_tests(verbose=False, coverage=False, markers=None):
    """Run unit tests"""
    cmd_parts = ["pytest", "tests/unit/"]

    if verbose:
        cmd_parts.append("-v")

    if coverage:
        cmd_parts.extend([
            "--cov=data_pipeline",
            "--cov=backend",
            "--cov-report=html",
            "--cov-report=term"
        ])

    if markers:
        cmd_parts.extend(["-m", markers])

    cmd = " ".join(cmd_parts)
    return run_command(cmd, "Running unit tests")


def run_integration_tests(verbose=False, markers=None):
    """Run integration tests"""
    cmd_parts = ["pytest", "tests/integration/", "-m", "integration"]

    if verbose:
        cmd_parts.append("-v")

    if markers:
        cmd_parts.extend(["-m", f"integration and ({markers})"])

    cmd = " ".join(cmd_parts)
    return run_command(cmd, "Running integration tests")


def run_specific_tests(test_pattern, verbose=False):
    """Run specific test patterns"""
    cmd_parts = ["pytest", "-k", test_pattern]

    if verbose:
        cmd_parts.append("-v")

    cmd = " ".join(cmd_parts)
    return run_command(cmd, f"Running tests matching: {test_pattern}")


def run_component_tests(component, verbose=False):
    """Run tests for specific component"""
    component_map = {
        "embedder": "tests/unit/test_bge_embedder.py",
        "chunker": "tests/unit/test_document_chunker.py",
        "indexer": "tests/unit/test_es_indexer.py",
        "ingestion": "tests/unit/test_ingest_papers.py",
        "search": "tests/unit/test_search_service.py"
    }

    if component not in component_map:
        print(f"❌ Unknown component: {component}")
        print(f"Available components: {', '.join(component_map.keys())}")
        return False

    cmd_parts = ["pytest", component_map[component]]

    if verbose:
        cmd_parts.append("-v")

    cmd = " ".join(cmd_parts)
    return run_command(cmd, f"Running {component} tests")


def run_all_tests(verbose=False, coverage=False, integration=False):
    """Run all tests"""
    success = True

    # Run unit tests
    success &= run_unit_tests(verbose=verbose, coverage=coverage)

    # Run integration tests if requested and services are available
    if integration:
        es_ok, minio_ok = check_services()
        if es_ok and minio_ok:
            success &= run_integration_tests(verbose=verbose)
        else:
            print("⚠️  Skipping integration tests - services not available")
            print("   Start services with: docker-compose up -d")

    return success


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Paper Search Engine Test Runner")

    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all", "specific", "component"],
        default="unit",
        help="Type of tests to run"
    )

    parser.add_argument(
        "--component",
        choices=["embedder", "chunker", "indexer", "ingestion", "search"],
        help="Specific component to test (use with --type component)"
    )

    parser.add_argument(
        "--pattern",
        help="Test name pattern to match (use with --type specific)"
    )

    parser.add_argument(
        "--markers",
        help="Pytest markers to filter tests (e.g., 'slow', 'embedder')"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--no-deps-check",
        action="store_true",
        help="Skip dependency checking"
    )

    parser.add_argument(
        "--no-services-check",
        action="store_true",
        help="Skip services checking for integration tests"
    )

    args = parser.parse_args()

    print("🧪 Paper Search Engine Test Runner")
    print("=" * 50)

    # Check dependencies
    if not args.no_deps_check:
        if not check_dependencies():
            sys.exit(1)

    # Run tests based on type
    success = False

    if args.type == "unit":
        success = run_unit_tests(
            verbose=args.verbose,
            coverage=args.coverage,
            markers=args.markers
        )

    elif args.type == "integration":
        if not args.no_services_check:
            es_ok, minio_ok = check_services()
            if not (es_ok and minio_ok):
                print("❌ Services not available for integration tests")
                print("Start services with: docker-compose up -d")
                sys.exit(1)

        success = run_integration_tests(
            verbose=args.verbose,
            markers=args.markers
        )

    elif args.type == "all":
        success = run_all_tests(
            verbose=args.verbose,
            coverage=args.coverage,
            integration=not args.no_services_check
        )

    elif args.type == "specific":
        if not args.pattern:
            print("❌ --pattern required for specific tests")
            sys.exit(1)

        success = run_specific_tests(
            test_pattern=args.pattern,
            verbose=args.verbose
        )

    elif args.type == "component":
        if not args.component:
            print("❌ --component required for component tests")
            sys.exit(1)

        success = run_component_tests(
            component=args.component,
            verbose=args.verbose
        )

    # Print summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()