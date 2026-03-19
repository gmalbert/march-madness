"""
download_haslametrics.py
------------------------
Scrapes the Offense and Defense rating tables from https://haslametrics.com/ratings.php
and saves them to:
  data_files/haslametrics_offense.csv
  data_files/haslametrics_defense.csv
  data_files/haslametrics_ratings.csv  (merged offense + defense)

Uses a SINGLE Selenium session to avoid repeated connection setup that triggers
rate-limiting.  The site requires JavaScript so requests/BeautifulSoup won't work.

Usage:
  python download_haslametrics.py
"""

import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data_files")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://haslametrics.com/ratings.php"

# The table on this page always has these 21 columns in order.
# Extra columns produced by the HTML parser are discarded.
OFFENSE_COLS = [
    "Rk", "Team", "Eff", "FTAR", "FT%", "FGAR", "FG%",
    "3PAR", "3P%", "MRAR", "MR%", "NPAR", "NP%",
    "PPSt", "PPSC", "SCC%", "%3PA", "%MRA", "%NPA", "Prox", "AP%",
]
DEFENSE_COLS = [
    "Rk", "Team", "Eff", "FTAR", "FT%", "FGAR", "FG%",
    "3PAR", "3P%", "MRAR", "MR%", "NPAR", "NP%",
    "PPSt", "PPSC", "SCC%", "%3PA", "%MRA", "%NPA", "Prox", "AP%",
]
N_COLS = len(OFFENSE_COLS)  # 21


# ---------------------------------------------------------------------------
# Build Selenium driver
# ---------------------------------------------------------------------------

def _build_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    opts.add_argument("--window-size=1920,1080")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ---------------------------------------------------------------------------
# Parse the ratings table from the current page source
# ---------------------------------------------------------------------------

def _parse_page(page_source: str, col_names: list, label: str) -> "pd.DataFrame | None":
    """
    Use pd.read_html to find the largest ratings table on the page,
    then trim to the first N_COLS columns and clean up rows.
    """
    try:
        tables = pd.read_html(StringIO(page_source), flavor="lxml")
    except Exception as exc:
        print(f"  [{label}] read_html failed: {exc}")
        return None

    if not tables:
        print(f"  [{label}] no tables found")
        return None

    print(f"  [{label}] found {len(tables)} HTML tables on page")

    # Find the table that has the most rows where col[0] is a valid integer rank.
    # The ratings table typically has 300+ such rows; other tables are much smaller.
    best: pd.DataFrame | None = None
    best_n = 0

    for tbl in tables:
        if tbl.shape[1] < 5:
            continue
        # Flatten multi-level columns
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [" ".join(str(c) for c in col).strip() for col in tbl.columns]
        # Count rows with a valid integer in first column
        rk_col = pd.to_numeric(tbl.iloc[:, 0], errors="coerce")
        n_valid = rk_col.notna().sum()
        if n_valid > best_n:
            best_n = n_valid
            best = tbl

    if best is None or best_n < 10:
        # Fallback: just pick the table with the most rows
        best = max(tables, key=lambda t: len(t))
        best_n = len(best)

    print(f"  [{label}] best table: {best_n} valid-rank rows, {best.shape[1]} raw columns")

    df = best.copy()

    # Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]

    # Trim to first N_COLS columns and assign canonical names
    df = df.iloc[:, :N_COLS]
    if len(df.columns) < N_COLS:
        # Pad if somehow fewer columns
        for i in range(N_COLS - len(df.columns)):
            df[f"_pad{i}"] = float("nan")
    df.columns = col_names

    # Keep only rows with a valid numeric Rk
    df["Rk"] = pd.to_numeric(df["Rk"], errors="coerce")
    df = df[df["Rk"].notna()].copy()

    # Keep only rows with a non-empty Team string
    df["Team"] = df["Team"].astype(str).str.strip()
    # Strip win-loss record appended to team name e.g. "Duke (31-3)" -> "Duke"
    df["Team"] = df["Team"].str.replace(r"\s*\(\d+-\d+\)\s*$", "", regex=True).str.strip()
    df = df[df["Team"].str.len() > 1]

    # Coerce all non-Team/Rk columns to numeric
    for col in col_names:
        if col not in ("Rk", "Team"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Rk").reset_index(drop=True)
    print(f"  [{label}] {len(df)} teams, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Select offense/defense in the dropdown
# ---------------------------------------------------------------------------

def _select_view(driver, target: str) -> bool:
    """
    Find the 'Select a rating set' dropdown and choose offense or defense.
    Matches on the option *value* attribute (case-insensitive) since some
    options have invisible label text.
    Returns True if the selection was made.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select

    try:
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel_el in selects:
            sel = Select(sel_el)
            for opt in sel.options:
                opt_text  = opt.text.strip().lower()
                opt_value = opt.get_attribute("value").strip().lower()
                if target in opt_text or target in opt_value:
                    val = opt.get_attribute("value")
                    print(f"  Selecting by value '{val}'")
                    sel.select_by_value(val)
                    return True
    except Exception as exc:
        print(f"  Dropdown selection failed: {exc}")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_haslametrics():
    print("=" * 60)
    print("Downloading Haslametrics Ratings")
    print(f"  Source: {BASE_URL}")
    print("=" * 60)

    driver = _build_driver()
    results = {}

    try:
        print("\nOpening page…")
        driver.get(BASE_URL)
        time.sleep(10)  # let the full table render before reading

        # ----------------------------------------------------------------
        # OFFENSE (default view)
        # ----------------------------------------------------------------
        print("\n--- OFFENSE ---")
        # Try to explicitly select 'Offense' to be safe
        _select_view(driver, "offense")
        time.sleep(5)

        off_df = _parse_page(driver.page_source, OFFENSE_COLS, "offense")
        if off_df is not None and not off_df.empty:
            out = OUT_DIR / "haslametrics_offense.csv"
            off_df.to_csv(out, index=False)
            print(f"  Saved → {out}")
            results["offense"] = off_df
        else:
            print("  WARNING: Could not parse offense table.")
            results["offense"] = None

        # ----------------------------------------------------------------
        # DEFENSE
        # ----------------------------------------------------------------
        print("\n--- DEFENSE ---")
        switched = _select_view(driver, "defense")
        if not switched:
            print("  Could not find defense option in dropdown — trying 'def'")
            switched = _select_view(driver, "def")
        time.sleep(4)  # wait for the table to reload

        def_df = _parse_page(driver.page_source, DEFENSE_COLS, "defense")
        if def_df is not None and not def_df.empty:
            out = OUT_DIR / "haslametrics_defense.csv"
            def_df.to_csv(out, index=False)
            print(f"  Saved → {out}")
            results["defense"] = def_df
        else:
            print("  WARNING: Could not parse defense table.")
            results["defense"] = None

    finally:
        driver.quit()

    # ----------------------------------------------------------------
    # Merge offense + defense into a single file
    # ----------------------------------------------------------------
    off_df = results.get("offense")
    def_df = results.get("defense")

    if off_df is not None and def_df is not None:
        off_m = off_df.rename(columns={c: f"O_{c}" for c in off_df.columns if c not in ("Team", "Rk")})
        def_m = def_df.rename(columns={
            **{c: f"D_{c}" for c in def_df.columns if c not in ("Team", "Rk")},
            "Rk": "D_Rk",
        })
        merged = off_m.merge(def_m.drop(columns=["D_Rk"], errors="ignore"), on="Team", how="outer")
        out = OUT_DIR / "haslametrics_ratings.csv"
        merged.to_csv(out, index=False)
        print(f"\nMerged offense+defense → {out}  ({len(merged)} teams)")
    elif off_df is not None:
        off_df.to_csv(OUT_DIR / "haslametrics_ratings.csv", index=False)
        print("\nOnly offense available — saved as haslametrics_ratings.csv")

    print("\nDone.")
    return results


if __name__ == "__main__":
    download_haslametrics()

