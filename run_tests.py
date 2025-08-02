#!/usr/bin/env python3
"""
Test runner script for the Daily Household Electricity Consumption Predictor.

This script runs all tests and provides a summary of results.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """Run all tests and return the result."""
    print("🧪 Running Daily Household Electricity Consumption Predictor Tests")
    print("=" * 70)

    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Run pytest with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--verbose",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml",
        "tests/",
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_unit_tests():
    """Run only unit tests."""
    print("🧪 Running Unit Tests")
    print("=" * 40)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--verbose",
        "--tb=short",
        "-m",
        "unit",
        "tests/",
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return False


def run_integration_tests():
    """Run only integration tests."""
    print("🧪 Running Integration Tests")
    print("=" * 40)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--verbose",
        "--tb=short",
        "-m",
        "integration",
        "tests/",
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running integration tests: {e}")
        return False


def main():
    """Main function to run tests based on command line arguments."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "unit":
            success = run_unit_tests()
        elif test_type == "integration":
            success = run_integration_tests()
        else:
            print(f"❌ Unknown test type: {test_type}")
            print("Available options: unit, integration, all (default)")
            return 1
    else:
        success = run_tests()

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
