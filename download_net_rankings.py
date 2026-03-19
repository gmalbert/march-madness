"""
download_net_rankings.py
------------------------
Scrapes the NCAA NET Rankings table from:
  https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings

The page is server-rendered HTML so plain requests + BeautifulSoup is sufficient.

Output:
  data_files/net_rankings.csv        — raw data with canonical_team column
  data_files/net_to_espn_matches.csv — team name mapping (created on first run)

Usage:
  python download_net_rankings.py
"""

import sys
import time
from difflib import get_close_matches
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUT_DIR = Path("data_files")
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Known name differences between NCAA.com and our canonical (ESPN/KenPom) names.
# Key = NCAA name, Value = canonical name.
OVERRIDES = {
    "Connecticut": "UConn",
    "Miami (FL)": "Miami",
    "Miami (OH)": "Miami (OH)",
    "NC State": "NC State",
    "Ole Miss": "Ole Miss",
    "LSU": "LSU",
    "USC": "USC",
    "Iowa St.": "Iowa State",
    "Michigan St.": "Michigan State",
    "Kansas St.": "Kansas State",
    "Ohio St.": "Ohio State",
    "Penn St.": "Penn State",
    "Texas A&M": "Texas A&M",
    "Florida St.": "Florida State",
    "Oklahoma St.": "Oklahoma State",
    "Arizona St.": "Arizona State",
    "Colorado St.": "Colorado State",
    "Oregon St.": "Oregon State",
    "Washington St.": "Washington State",
    "Wichita St.": "Wichita State",
    "Utah St.": "Utah State",
    "New Mexico St.": "New Mexico State",
    "Idaho St.": "Idaho State",
    "Sam Houston St.": "Sam Houston State",
    "Sacramento St.": "Sacramento State",
    "South Dakota St.": "South Dakota State",
    "North Dakota St.": "North Dakota State",
    "McNeese St.": "McNeese State",
    "Kennesaw St.": "Kennesaw State",
    "Nicholls St.": "Nicholls State",
    "Murray St.": "Murray State",
    "Morehead St.": "Morehead State",
    "Jacksonville St.": "Jacksonville State",
    "Grambling St.": "Grambling State",
    "Norfolk St.": "Norfolk State",
    "Jackson St.": "Jackson State",
    "Morgan St.": "Morgan State",
    "Coppin St.": "Coppin State",
    "Alcorn St.": "Alcorn State",
    "Alcorn": "Alcorn State",
    "Delaware St.": "Delaware State",
    "Savannah St.": "Savannah State",
    "Chicago St.": "Chicago State",
    "Fort Valley St.": "Fort Valley State",
    "Southeast Missouri St.": "Southeast Missouri State",
    "Missouri St.": "Missouri State",
    "Youngstown St.": "Youngstown State",
    "Bowling Green St.": "Bowling Green",
    "Ball St.": "Ball State",
    "Kent St.": "Kent State",
    "Georgia St.": "Georgia State",
    "Boise St.": "Boise State",
    "San Jose St.": "San Jose State",
    "San Diego St.": "San Diego State",
    "Long Beach St.": "Long Beach State",
    "Cal St. Bakersfield": "Cal State Bakersfield",
    "Cal St. Fullerton": "Cal State Fullerton",
    "Cal St. Northridge": "Cal State Northridge",
    "SIU Edwardsville": "SIU Edwardsville",
    "SIUE": "SIU Edwardsville",
    "Army West Point": "Army",
    "BYU": "BYU",
    "LIU": "LIU",
    "UMBC": "UMBC",
    "UTEP": "UTEP",
    "UTSA": "UTSA",
    "SMU": "SMU",
    "VCU": "VCU",
    "UCF": "UCF",
    "UAB": "UAB",
    "UNLV": "UNLV",
    "UMKC": "UMKC",
    "UIC": "Illinois-Chicago",
    "UT Rio Grande Valley": "UT Rio Grande Valley",
    "UTRGV": "UT Rio Grande Valley",
    "UNI": "Northern Iowa",
    "UIW": "Incarnate Word",
    "FDU": "Fairleigh Dickinson",
    "NIU": "Northern Illinois",
    "USC": "USC",
    "Southern California": "USC",
    "App State": "Appalachian State",
    "Appalachian St.": "Appalachian State",
    "Ga. Southern": "Georgia Southern",
    "Southern U.": "Southern",
    "Southeast Mo. St.": "Southeast Missouri State",
    "Lamar University": "Lamar",
    "Prairie View": "Prairie View",
    "CSU Bakersfield": "Cal State Bakersfield",
    "North Ala.": "North Alabama",
    "Northern Colo.": "Northern Colorado",
    "Northern Ky.": "Northern Kentucky",
    "Northern Ariz.": "Northern Arizona",
    "Eastern Wash.": "Eastern Washington",
    "Eastern Mich.": "Eastern Michigan",
    "Eastern Ky.": "Eastern Kentucky",
    "Eastern Ill.": "Eastern Illinois",
    "Western Ky.": "Western Kentucky",
    "Western Mich.": "Western Michigan",
    "Western Caro.": "Western Carolina",
    "Western Ill.": "Western Illinois",
    "Southern Ill.": "Southern Illinois",
    "Southern Miss.": "Southern Miss",
    "Southern Ind.": "Southern Indiana",
    "Central Mich.": "Central Michigan",
    "Central Ark.": "Central Arkansas",
    "Central Conn. St.": "Central Connecticut State",
    "Middle Tenn.": "Middle Tennessee",
    "Fla. Atlantic": "Florida Atlantic",
    "Cal St. Fullerton": "Cal State Fullerton",
    "Cal St. Bakersfield": "Cal State Bakersfield",
    "Charleston So.": "Charleston Southern",
    "South Fla.": "South Florida",
    "St. Thomas (MN)": "St. Thomas",
    "St. John's (NY)": "St. John's",
    "Southeastern La.": "Southeastern Louisiana",
    "A&M-Corpus Christi": "Texas A&M-Corpus Christi",
    "Mississippi Val.": "Mississippi Valley State",
    "West Ga.": "West Georgia",
    "South Carolina St.": "South Carolina State",
    "Morgan St.": "Morgan State",
    "Jackson St.": "Jackson State",
    "Coppin St.": "Coppin State",
    "Delaware St.": "Delaware State",
    "UL Lafayette": "Louisiana",
    "Louisiana-Lafayette": "Louisiana",
    "Louisiana Monroe": "UL Monroe",
    "Louisiana Tech": "Louisiana Tech",
    "FIU": "FIU",
    "FAU": "Florida Atlantic",
    "Florida Atlantic": "Florida Atlantic",
    "Alabama A&M": "Alabama A&M",
    "Bethune-Cookman": "Bethune-Cookman",
    "Prairie View A&M": "Prairie View",
    "Maryland-Eastern Shore": "Maryland-Eastern Shore",
    "Loyola Chicago": "Loyola Chicago",
    "Loyola Maryland": "Loyola Maryland",
    "Loyola (MD)": "Loyola Maryland",
    "Loyola (IL)": "Loyola Chicago",
    "Mount St. Mary's": "Mount St. Mary's",
    "Saint Mary's (CA)": "Saint Mary's",
    "Saint Joseph's": "Saint Joseph's",
    "Saint Peter's": "Saint Peter's",
    "Appalachian St.": "Appalachian State",
    "Georgia Southern": "Georgia Southern",
    "Coastal Carolina": "Coastal Carolina",
    "South Alabama": "South Alabama",
    "Old Dominion": "Old Dominion",
    "James Madison": "James Madison",
    "Stony Brook": "Stony Brook",
    "Central Connecticut": "Central Connecticut State",
    "CCSU": "Central Connecticut State",
    "Southern Utah": "Southern Utah",
    "California Baptist": "California Baptist",
    "Robert Morris": "Robert Morris",
    "Queens (NC)": "Queens",
    "UT Arlington": "UT Arlington",
    "Tennessee-Martin": "UT Martin",
    "Tennessee Martin": "UT Martin",
    "Western Carolina": "Western Carolina",
    "UNC Greensboro": "UNC Greensboro",
    "UNC Asheville": "UNC Asheville",
    "UNC Wilmington": "UNC Wilmington",
    "NC A&T": "North Carolina A&T",
    "North Carolina A&T": "North Carolina A&T",
    "NC Central": "NC Central",
    "Hampton": "Hampton",
    "Howard": "Howard",
    "Morgan State": "Morgan State",
    "Drexel": "Drexel",
    "Towson": "Towson",
    "Elon": "Elon",
    "William & Mary": "William & Mary",
    "Charleston": "Charleston",
    "Col. of Charleston": "Charleston",
    "College of Charleston": "Charleston",
    "Abilene Christian": "Abilene Christian",
    "Stephen F. Austin": "Stephen F. Austin",
    "Incarnate Word": "Incarnate Word",
    "Central Arkansas": "Central Arkansas",
    "Texas State": "Texas State",
    "Southern Miss.": "Southern Miss",
    "Southern Mississippi": "Southern Miss",
    "East Carolina": "East Carolina",
    "UTSA": "UTSA",
    "Rice": "Rice",
    "Charlotte": "Charlotte",
    "North Texas": "North Texas",
    "Massachusetts": "Massachusetts",
    "UMass": "Massachusetts",
    "Rhode Island": "Rhode Island",
    "URI": "Rhode Island",
    "George Washington": "George Washington",
    "La Salle": "La Salle",
    "Fordham": "Fordham",
    "Manhattan": "Manhattan",
    "Pittsburgh": "Pittsburgh",
    "Pitt": "Pittsburgh",
    "Seton Hall": "Seton Hall",
    "DePaul": "DePaul",
    "Providence": "Providence",
    "St. John's (NY)": "St. John's",
    "St. John's": "St. John's",
    "Xavier": "Xavier",
    "Villanova": "Villanova",
    "Marquette": "Marquette",
    "Georgetown": "Georgetown",
    "Butler": "Butler",
    "Creighton": "Creighton",
    "Indiana": "Indiana",
    "Purdue Fort Wayne": "Purdue Fort Wayne",
    "IUPUI": "IUPUI",
    "Southern Indiana": "Southern Indiana",
    "Wofford": "Wofford",
    "The Citadel": "The Citadel",
    "Mercer": "Mercer",
    "Samford": "Samford",
    "Furman": "Furman",
    "VMI": "VMI",
    "East Tennessee St.": "East Tennessee State",
    "ETSU": "East Tennessee State",
    "Chattanooga": "Chattanooga",
    "UNCG": "UNC Greensboro",
    "USC Upstate": "USC Upstate",
    "Presbyterian": "Presbyterian",
    "Gardner-Webb": "Gardner-Webb",
    "High Point": "High Point",
    "Longwood": "Longwood",
    "Norfolk State": "Norfolk State",
    "Hampton": "Hampton",
    "SC State": "South Carolina State",
    "South Carolina St.": "South Carolina State",
    "Stetson": "Stetson",
    "Bellarmine": "Bellarmine",
    "North Florida": "North Florida",
    "Jacksonville": "Jacksonville",
    "FGCU": "Florida Gulf Coast",
    "Florida Gulf Coast": "Florida Gulf Coast",
    "Lipscomb": "Lipscomb",
    "Belmont": "Belmont",
    "Eastern Kentucky": "Eastern Kentucky",
    "Morehead State": "Morehead State",
    "Murray State": "Murray State",
    "Tennessee St.": "Tennessee State",
    "Tennessee State": "Tennessee State",
    "Austin Peay": "Austin Peay",
    "Southern Illinois": "Southern Illinois",
    "Illinois State": "Illinois State",
    "Bradley": "Bradley",
    "Missouri State": "Missouri State",
    "Northern Iowa": "Northern Iowa",
    "Drake": "Drake",
    "Indiana State": "Indiana State",
    "Valparaiso": "Valparaiso",
    "Evansville": "Evansville",
    "Loyola-Chicago": "Loyola Chicago",
    "Notre Dame": "Notre Dame",
    "Boston College": "Boston College",
    "Syracuse": "Syracuse",
    "Clemson": "Clemson",
    "Wake Forest": "Wake Forest",
    "Georgia Tech": "Georgia Tech",
    "Virginia Tech": "Virginia Tech",
    "Louisville": "Louisville",
    "Miami": "Miami",
    "North Carolina": "North Carolina",
    "Duke": "Duke",
    "Virginia": "Virginia",
    "Pittsburgh": "Pittsburgh",
    "Minnesota": "Minnesota",
    "Michigan": "Michigan",
    "Ohio State": "Ohio State",
    "Indiana": "Indiana",
    "Illinois": "Illinois",
    "Purdue": "Purdue",
    "Northwestern": "Northwestern",
    "Rutgers": "Rutgers",
    "Maryland": "Maryland",
    "Wisconsin": "Wisconsin",
    "Nebraska": "Nebraska",
    "Penn State": "Penn State",
    "Iowa": "Iowa",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_canonical_reference() -> list[str]:
    """Load all known canonical names from existing mapping files."""
    names = set()

    kp_map = OUT_DIR / "kenpom_to_espn_matches.csv"
    if kp_map.exists():
        df = pd.read_csv(kp_map)
        for col in df.columns:
            names.update(df[col].dropna().str.replace(r"\s*\(fuzzy\)\s*$", "", regex=True).tolist())

    bart_map = OUT_DIR / "bart_to_espn_matches.csv"
    if bart_map.exists():
        df = pd.read_csv(bart_map)
        for col in df.columns:
            names.update(df[col].dropna().str.replace(r"\s*\(fuzzy\)\s*$", "", regex=True).tolist())

    kp_rat = OUT_DIR / "kenpom_ratings.csv"
    if kp_rat.exists():
        df = pd.read_csv(kp_rat)
        names.update(df["Team"].dropna().tolist())

    return sorted(names)


def _canonical_name(raw: str, canon_names: list[str]) -> str:
    """Map a raw NCAA.com team name to a canonical name."""
    raw = raw.strip()

    # Direct override
    if raw in OVERRIDES:
        return OVERRIDES[raw]

    # Exact match in canonical list
    if raw in canon_names:
        return raw

    # Fuzzy match (high confidence)
    matches = get_close_matches(raw, canon_names, n=1, cutoff=0.82)
    if matches:
        return matches[0]

    # Looser fuzzy match
    matches = get_close_matches(raw, canon_names, n=1, cutoff=0.65)
    if matches:
        return matches[0]

    # No match — keep original
    return raw


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def download_net_rankings() -> pd.DataFrame:
    print("=" * 60)
    print("Downloading NCAA NET Rankings")
    print(f"  Source: {URL}")
    print("=" * 60)

    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("No <table> found on the NCAA rankings page.")

    # Parse via pd.read_html for robustness
    try:
        dfs = pd.read_html(StringIO(str(table)), flavor="lxml")
        df = dfs[0]
    except Exception:
        # Manual parse fallback
        rows = []
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        for tr in table.find_all("tr")[1:]:
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if tds:
                rows.append(tds)
        df = pd.DataFrame(rows, columns=headers[:len(rows[0])] if rows else headers)

    # Normalize column names
    df.columns = df.columns.str.strip()
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # Expected columns: Rank, School, Record, Conf, Road, Neutral, Home, Non-Div I, Prev,
    #                   Quad 1, Quad 2, Quad 3, Quad 4
    # Rename for clarity
    rename_map = {
        "School": "Team",
        "Rank": "NET_Rank",
        "Prev": "NET_Prev",
        "Record": "Record",
        "Conf": "Conf",
        "Road": "Road",
        "Neutral": "Neutral",
        "Home": "Home",
        "Non-Div I": "Non_DivI",
        "Quad 1": "Quad1",
        "Quad 2": "Quad2",
        "Quad 3": "Quad3",
        "Quad 4": "Quad4",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Coerce numeric columns
    for col in ["NET_Rank", "NET_Prev"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without a team name
    if "Team" in df.columns:
        df = df[df["Team"].notna() & (df["Team"].str.strip() != "")].copy()

    # Parse wins/losses from Record
    if "Record" in df.columns:
        rec = df["Record"].str.extract(r"(\d+)[–\-](\d+)")
        if rec is not None and rec.shape[1] == 2:
            df["Wins"] = pd.to_numeric(rec[0], errors="coerce")
            df["Losses"] = pd.to_numeric(rec[1], errors="coerce")

    # Build canonical mapping
    canon_names = _build_canonical_reference()
    mapping_rows = []
    seen_canon = {}
    for raw in df["Team"].tolist():
        canon = _canonical_name(raw, canon_names)
        mapping_rows.append({"ncaa_net": raw, "espn_match": canon})
        seen_canon[raw] = canon

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(OUT_DIR / "net_to_espn_matches.csv", index=False)
    print(f"  Saved mapping → data_files/net_to_espn_matches.csv")

    df["canonical_team"] = df["Team"].map(seen_canon)

    # Sort by rank
    if "NET_Rank" in df.columns:
        df = df.sort_values("NET_Rank").reset_index(drop=True)

    out = OUT_DIR / "net_rankings.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  ({len(df)} teams)")

    # Quick preview
    preview_cols = [c for c in ["NET_Rank", "canonical_team", "Record", "Conf",
                                 "Quad1", "Quad2", "NET_Prev"] if c in df.columns]
    print(df[preview_cols].head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    download_net_rankings()
