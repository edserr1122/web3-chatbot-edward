"""
Main entry point for Crypto Analysis Chatbot.
Run this file to start the CLI interface.

Usage:
    python main.py           # Production mode (only errors in console, warnings/errors in log file)
    python main.py --verbose # Development mode (all logs in console)
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interfaces.cli import run_cli


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Analysis Chatbot - AI-powered cryptocurrency analysis assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py           # Run in production mode (only errors in console, all logs in file)
  python main.py --verbose # Run in development mode (all logs in console and file)
        """
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (shows INFO/WARNING/ERROR in console, default: only ERROR)'
    )
    parser.add_argument(
        '-s', '--session',
        type=str,
        help='Optional session ID to resume (default: new session)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(verbose=args.verbose, session_id=args.session)

