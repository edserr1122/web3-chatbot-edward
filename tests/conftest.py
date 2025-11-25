"""
Pytest configuration and shared fixtures.
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Groq API Key
API_KEY = 'gsk_abcd'

# Set test environment variables
os.environ.setdefault('GROQ_API_KEY', API_KEY)
os.environ.setdefault('LOG_LEVEL', 'ERROR')  # Reduce noise in tests
