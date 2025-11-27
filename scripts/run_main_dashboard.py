#!/usr/bin/env python3
"""
Script to run the main dashboard.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.dashboard.main import CyberattackDashboard

def main():
    """Run the main dashboard."""
    dashboard = CyberattackDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
