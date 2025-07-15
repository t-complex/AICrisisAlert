#!/usr/bin/env python3
"""
Comprehensive test runner for AICrisisAlert project.
Runs all tests including unit, integration, and API tests.

This script provides a unified interface for running all project tests and code quality checks.
It supports running specific test types, generating coverage reports, and performing linting.

USAGE EXAMPLES:
    # Run all tests and checks (default behavior)
    python scripts/run_all_tests.py
    
    # Run only unit tests with verbose output
    python scripts/run_all_tests.py --type unit --verbose
    
    # Run only integration tests with coverage report
    python scripts/run_all_tests.py --type integration --coverage
    
    # Run only API tests
    python scripts/run_all_tests.py --type api
    
    # Run only linting checks
    python scripts/run_all_tests.py --lint
    
    # Check for unused imports
    python scripts/run_all_tests.py --imports
    
    # Run everything with coverage and verbose output
    python scripts/run_all_tests.py --all --coverage --verbose

REQUIREMENTS:
    - pytest: pip install pytest
    - flake8: pip install flake8 (for linting)
    - autoflake: pip install autoflake (for import checks)
    - pytest-cov: pip install pytest-cov (for coverage reports)

EXIT CODES:
    - 0: All checks passed successfully
    - 1: One or more checks failed
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests(test_type=None, verbose=False, coverage=False):
    """
    Run tests based on type specified.
    
    Args:
        test_type: 'unit', 'integration', 'api', or None for all
        verbose: Enable verbose output
        coverage: Generate coverage report
    """
    tests_dir = project_root / "tests"
    
    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        return False
    
    # Build pytest command
    cmd = ["python3", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add test discovery patterns
    if test_type:
        if test_type == "unit":
            cmd.append(str(tests_dir / "unit"))
        elif test_type == "integration":
            cmd.append(str(tests_dir / "integration"))
        elif test_type == "api":
            cmd.append(str(tests_dir / "api"))
        else:
            print(f"❌ Unknown test type: {test_type}")
            return False
    else:
        # Run all tests
        cmd.append(str(tests_dir))
    
    print(f"🚀 Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("=" * 60)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"❌ Tests failed with exit code: {e.returncode}")
        return False

def run_linting():
    """Run code linting checks."""
    print("🔍 Running code linting...")
    print("=" * 60)
    
    # Check for flake8
    try:
        subprocess.run(["python3", "-m", "flake8", "src", "tests"], 
                      cwd=project_root, check=True)
        print("✅ Linting passed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Linting failed!")
        return False
    except FileNotFoundError:
        print("⚠️  flake8 not found. Install with: pip install flake8")
        return False

def run_import_checks():
    """Check for unused imports and optimize imports."""
    print("🔍 Checking for unused imports...")
    print("=" * 60)
    
    # Check for autoflake
    try:
        subprocess.run(["python3", "-m", "autoflake", "--check", "--remove-all-unused-imports", 
                       "--recursive", "src", "tests"], 
                      cwd=project_root, check=True)
        print("✅ No unused imports found!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Unused imports found! Run: autoflake --in-place --remove-all-unused-imports --recursive src tests")
        return False
    except FileNotFoundError:
        print("⚠️  autoflake not found. Install with: pip install autoflake")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run all tests for AICrisisAlert")
    parser.add_argument("--type", choices=["unit", "integration", "api"], 
                       help="Run specific test type")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", 
                       help="Generate coverage report")
    parser.add_argument("--lint", action="store_true", 
                       help="Run linting checks")
    parser.add_argument("--imports", action="store_true", 
                       help="Check for unused imports")
    parser.add_argument("--all", action="store_true", 
                       help="Run all checks (tests, linting, imports)")
    
    args = parser.parse_args()
    
    success = True
    
    if args.all or (not args.type and not args.lint and not args.imports):
        # Run everything by default
        print("🎯 Running comprehensive test suite...")
        print("=" * 60)
        
        # Run linting
        if not run_linting():
            success = False
        
        # Check imports
        if not run_import_checks():
            success = False
        
        # Run tests
        if not run_tests(verbose=args.verbose, coverage=args.coverage):
            success = False
    
    else:
        # Run specific checks
        if args.lint:
            if not run_linting():
                success = False
        
        if args.imports:
            if not run_import_checks():
                success = False
        
        if args.type or not (args.lint or args.imports):
            if not run_tests(test_type=args.type, verbose=args.verbose, coverage=args.coverage):
                success = False
    
    print("=" * 60)
    if success:
        print("🎉 All checks completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 