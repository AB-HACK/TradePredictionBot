"""
Path Setup Utility
Centralizes sys.path configuration for imports
"""

import sys
import os
from pathlib import Path

def setup_paths():
    """
    Setup sys.path for importing project modules
    Should be called once at application startup
    """
    # Get project root (3 levels up from this file: backend/app/utils/)
    project_root = Path(__file__).parent.parent.parent.parent
    
    paths_to_add = [
        str(project_root),  # Project root
        str(project_root / 'quantitative_trading' / 'src'),  # Quantitative trading modules
        str(project_root / 'src'),  # Core src modules
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

# Auto-setup on import
setup_paths()
