#!/usr/bin/env python3
"""Generate canonical KenPom and BartTorvik datasets.

Usage: python scripts/generate_canonical_datasets.py
"""

import sys
from pathlib import Path
# ensure project root is on sys.path so local packages can be imported
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from data_tools.efficiency_loader import EfficiencyDataLoader


def main():
    loader = EfficiencyDataLoader()
    kp, bt = loader.save_canonical_datasets()
    print(f"Saved {len(kp)} KenPom teams and {len(bt)} BartTorvik teams")


if __name__ == '__main__':
    main()
