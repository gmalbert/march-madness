#!/usr/bin/env python3
"""Quick normalization tests for known problematic ESPN names.

Usage: python scripts/normalize_tests.py
"""

import sys
from pathlib import Path
# ensure project root is on sys.path so local packages can be imported
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import importlib
import predictions
importlib.reload(predictions)

TESTS = [
    'IUPUI',
    'IU Indianapolis Jaguars',
    'LIU',
    'LIU Sharks',
    'Long Island University Sharks',
    'New Haven Chargers',
    'New Haven',
    'Saint Francis Red Flash',
]

for t in TESTS:
    print(f"{t} -> {predictions.normalize_team_name(t)}")
