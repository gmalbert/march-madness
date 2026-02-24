#!/usr/bin/env python3
"""Utility to locate non-UTF8/odd bytes in GitHub workflow YAML files.

When workflows fail to parse on a runner because of an encoding error, this
script can be used locally to pinpoint the offending position and (optionally)
rewrite the file to UTF-8.

Usage:
    python scripts/check_yaml_encoding.py           # just report problems
    python scripts/check_yaml_encoding.py --fix    # rewrite files with UTF-8

The tool walks `.github/workflows/*.yml` and tests whether the file can be
loaded by PyYAML.  If loading fails with a codec error, it scans the file's
bytes to find the first non-ascii (>127) byte and shows its context.
"""

import argparse
import glob
import sys
from pathlib import Path

import yaml


def report_file(path: Path, fix: bool) -> bool:
    """Return True if the file is clean or fixed, False if error remains."""
    text = path.read_bytes()
    try:
        yaml.safe_load(text)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"{path}: YAML parse error: {exc}")
        # if it's a codec issue, locate the first offending byte
        for idx, b in enumerate(text):
            if b > 127:
                start = max(0, idx - 20)
                end = min(len(text), idx + 20)
                snippet = text[start:end]
                # show hex and printable
                pretty = ' '.join(f"{c:02X}" for c in snippet)
                printable = ''.join((chr(c) if 32 <= c < 127 else '.') for c in snippet)
                print(f"  first high byte at {idx} (0x{b:02X})")
                print(f"  context bytes: {pretty}")
                print(f"  context text : {printable}")
                break
        if fix:
            # attempt to re-encode as utf-8, replacing problematic bytes
            fixed = text.decode('utf-8', errors='replace')
            path.write_text(fixed, encoding='utf-8')
            print(f"  file rewritten to UTF-8 with replacements")
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description="Check GitHub workflow YAML encoding")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="rewrite files as utf-8 with replacements",
    )
    args = parser.parse_args()

    bad = False
    for p in glob.glob(".github/workflows/*.yml"):
        path = Path(p)
        ok = report_file(path, fix=args.fix)
        if not ok:
            bad = True
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
