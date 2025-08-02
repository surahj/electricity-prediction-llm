#!/usr/bin/env python3
"""
Test script for deployment verification
"""

import os
import sys


def test_imports():
    """Test if all required modules can be imported."""
    try:
        from src.app import ElectricityPredictorApp
        from src.model import ElectricityConsumptionModel
        from src.data_generator import DataGenerator

        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_app_creation():
    """Test if the app can be created."""
    try:
        from src.app import ElectricityPredictorApp

        app = ElectricityPredictorApp()
        print("✅ App creation successful")
        return True
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False


def test_interface_creation():
    """Test if the interface can be created."""
    try:
        from src.app import ElectricityPredictorApp

        app = ElectricityPredictorApp()
        interface = app.create_interface()
        print("✅ Interface creation successful")
        return True
    except Exception as e:
        print(f"❌ Interface creation error: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing deployment...")

    tests = [test_imports, test_app_creation, test_interface_creation]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please fix before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
