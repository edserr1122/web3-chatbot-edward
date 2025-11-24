"""
Main entry point for Crypto Analysis Chatbot.
Run this file to start the CLI interface.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interfaces.cli import run_cli


if __name__ == "__main__":
    run_cli()

