#!/usr/bin/env python3
"""
Hugging Face Spaces deployment entry point
This file is required for Hugging Face Spaces to recognize and run the app
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import and run the main app
from src.app import main

if __name__ == "__main__":
    main()
