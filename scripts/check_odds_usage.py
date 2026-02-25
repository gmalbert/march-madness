"""Utility script to report remaining Odds API tokens.

This script wraps the OpeningLineCapture helper so you can quickly check how
many requests have been used/are left without digging into the database.

Usage:
    python scripts/check_odds_usage.py            # just show current counters
    python scripts/check_odds_usage.py --refresh  # perform one API call and update counters

The script is intentionally lightweight; it uses the same logic as the
capture job but avoids writing any files.
"""

import argparse
import sys
import os

# ensure top-level module path
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('scripts'))

try:
    from auto_capture_opening_lines import OpeningLineCapture
except ImportError as e:
    print(f"Failed to import OpeningLineCapture: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Check remaining Odds API requests")
    parser.add_argument("--refresh", action="store_true", help="fetch fresh odds to update counters")
    args = parser.parse_args()

    cap = OpeningLineCapture()
    if args.refresh:
        print("Fetching current odds to refresh usage info...")
        cap.get_current_odds()

    print(f"Requests used:      {cap.requests_used}")
    print(f"Requests remaining: {cap.requests_remaining}")


if __name__ == "__main__":
    main()
