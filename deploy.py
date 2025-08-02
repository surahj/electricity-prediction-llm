#!/usr/bin/env python3
"""
Deployment script for Render deployment
"""

import os
import sys
from src.app import main

if __name__ == "__main__":
    # Set environment variables for production
    os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
    os.environ.setdefault("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860"))

    # Launch the application
    main()
