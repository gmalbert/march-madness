# Sports-Quant Scraping & Data Sources

> Source: [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) — March Madness module  
> Purpose: Document scraping techniques and new data sources we can adopt.

---

## Executive Summary

| Data Source | Our Current Approach | sports-quant Approach | Advantage |
|-------------|---------------------|----------------------|-----------|
| KenPom | Selenium → live page | requests + BeautifulSoup → Web Archive | No Selenium needed, historical data |
| BartTorvik | Selenium → CSV download | urllib → Time Machine JSON API | 13 stats, pre-tournament snapshots |
| BartTorvik Snapshots | Season-end only | Day-before-R64 snapshots | Prevents data leakage |
| ESPN Injuries | Not used | Scraper + LLM parser | New data dimension |
| Sports Reference | Not used | Player-level stats | Enables injury impact scoring |

---

## 1. KenPom Scraping — Web Archive Approach

### Our Current Method
We use **Selenium** (headless Chrome) to load `kenpom.com` and parse the HTML. This:
- Requires Chrome/ChromeDriver installed
- Is fragile (Selenium sessions fail silently)
- Can only fetch current-season data
- No historical snapshots

### sports-quant Method
Uses **requests + BeautifulSoup** against Web Archive snapshots — no Selenium required:

```python
# kenpom_scraper.py — Web Archive approach (no Selenium needed)

import re
import logging
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Web Archive URLs for historical pre-tournament KenPom snapshots
KENPOM_ARCHIVE_URLS = {
    2017: "https://web.archive.org/web/20170312131016/http://kenpom.com/",
    2018: "https://web.archive.org/web/20180311122559/https://kenpom.com/",
    2019: "https://web.archive.org/web/20190317211809/https://kenpom.com/",
    2021: "https://web.archive.org/web/20210318152437/https://kenpom.com/",
    2022: "https://web.archive.org/web/20220312213724/https://kenpom.com/",
    2023: "https://web.archive.org/web/20230314165625/https://kenpom.com/",
    2024: "https://web.archive.org/web/20240321081134/https://kenpom.com/",
    2025: "https://web.archive.org/web/20250314000625/https://kenpom.com/",
}

# Standard KenPom column names after parsing
KENPOM_COLUMNS = [
    "Rank", "Team", "Conf", "W-L", "AdjEM",
    "AdjO", "AdjO_Rank", "AdjD", "AdjD_Rank",
    "AdjT", "AdjT_Rank", "Luck", "Luck_Rank",
    "AdjEM_SOS", "AdjEM_SOS_Rank", "OppO", "OppO_Rank",
    "OppD", "OppD_Rank", "AdjEM_NCSOS", "AdjEM_NCSOS_Rank",
    "Year",
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3"
    )
}


def scrape_kenpom_archive(url: str, year: int) -> pd.DataFrame:
    """Fetch and parse a single KenPom Web Archive page.
    
    Uses requests + BeautifulSoup instead of Selenium.
    Web Archive URLs are stable and don't require JavaScript rendering.
    
    Args:
        url: Web Archive URL for KenPom page.
        year: Season year to tag data with.
        
    Returns:
        DataFrame of KenPom ratings, or empty DataFrame on error.
    """
    if year == 2020:
        logger.info("Skipping 2020 (no tournament)")
        return pd.DataFrame()
    
    try:
        page = requests.get(url, headers=REQUEST_HEADERS)
        page.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Request failed for %s: %s", url, e)
        return pd.DataFrame()
    
    try:
        soup = BeautifulSoup(page.text, features="lxml")
        table_full = soup.find_all("table", {"id": "ratings-table"})
        
        if not table_full:
            logger.warning("No ratings table at %s", url)
            return pd.DataFrame()
        
        # Remove thead elements that confuse pd.read_html
        thead = table_full[0].find_all("thead")
        table = table_full[0]
        for weird in thead:
            table = str(table).replace(str(weird), "")
        
        df = pd.read_html(StringIO(table))[0]
    except Exception as e:
        logger.error("HTML processing failed for %s: %s", url, e)
        return pd.DataFrame()
    
    df["Year"] = year
    return df


def scrape_all_kenpom(
    output_path: str = "data_files/kenpom_historical.csv",
) -> pd.DataFrame:
    """Scrape KenPom ratings for all available historical years.
    
    Returns:
        Combined DataFrame with standardized columns.
    """
    frames = []
    for year, url in sorted(KENPOM_ARCHIVE_URLS.items()):
        logger.info("Scraping KenPom year %d", year)
        df = scrape_kenpom_archive(url, year)
        if not df.empty:
            frames.append(df)
    
    if not frames:
        logger.warning("No KenPom data scraped")
        return pd.DataFrame()
    
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined.columns = KENPOM_COLUMNS
    
    # Clean team names: remove digits and semicolons
    combined["Team"] = combined["Team"].apply(
        lambda x: re.sub(r"\d", "", str(x)).strip().replace(";", "")
    )
    
    combined.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(combined), output_path)
    return combined
```

### Key Advantages
1. **No Selenium dependency** — uses plain `requests` library
2. **Historical data** — Web Archive has snapshots for every year back to 2017
3. **Pre-tournament snapshots** — URLs are from Selection Sunday week (before tournament games)
4. **Stable URLs** — Web Archive URLs don't change
5. **Reproducible** — same URL always returns same data

---

## 2. BartTorvik Time Machine API

### Our Current Method
We use **Selenium** to download a CSV from `barttorvik.com/team-tables_each.php?csv=1`. This:
- Gets season-end data (includes tournament results = **data leakage**)
- Only captures 2 stats (Adj OE, Adj DE)
- Requires Selenium

### sports-quant Method
Uses BartTorvik's **Time Machine** — a JSON API that provides daily historical snapshots:

```python
# barttorvik_time_machine.py — Full implementation

import gzip
import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Snapshot dates: day before R64 starts for each tournament year
# Captures First Four results but excludes main-draw games
SNAPSHOT_DATES = {
    2010: "20100314", 2011: "20110313", 2012: "20120311",
    2013: "20130316", 2014: "20140316", 2015: "20150315",
    2016: "20160313", 2017: "20170312", 2018: "20180311",
    2019: "20190317",
    # 2020: no tournament
    2021: "20210314", 2022: "20220313", 2023: "20230312",
    2024: "20240317", 2025: "20250316",
}

TIME_MACHINE_URL = (
    "https://barttorvik.com/timemachine/team_results/"
    "{date}_team_results.json.gz"
)

# Column indices in the Time Machine JSON arrays
# The JSON returns lists of lists (no headers) — these are positional indices
COL_IDX = {
    "Team": 1,
    "Bart_Rank": 0,
    "Bart_AdjOE": 4,        # Adjusted Offensive Efficiency
    "Bart_AdjDE": 6,        # Adjusted Defensive Efficiency
    "Bart_Barthag": 8,      # Power rating (probability of beating average team)
    "Bart_AdjT": 44,        # Adjusted Tempo
    "Bart_SOS": 15,         # Strength of Schedule
    "Bart_NCSOS": 16,       # Non-Conference SOS
    "Bart_EliteSOS": 21,    # Elite SOS (quality of top opponents)
    "Bart_WAB": 41,         # Wins Above Bubble
    "Bart_QualO": 29,       # Offense rating vs quality opponents
    "Bart_QualD": 30,       # Defense rating vs quality opponents
    "Bart_QualBarthag": 31, # Overall rating vs quality opponents
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def download_barttorvik_snapshot(year: int) -> pd.DataFrame:
    """Download pre-tournament BartTorvik snapshot from Time Machine.
    
    IMPORTANT: Uses day-before-R64 snapshots to prevent data leakage.
    The standard BartTorvik CSV download includes tournament results
    and MUST NOT be used for tournament prediction models.
    
    Args:
        year: Season year (e.g. 2025 for 2024-25 season).
        
    Returns:
        DataFrame with 13 BartTorvik columns + Team + Year, or empty on error.
    """
    date_str = SNAPSHOT_DATES.get(year)
    if date_str is None:
        logger.warning("No snapshot date for year %d", year)
        return pd.DataFrame()
    
    url = TIME_MACHINE_URL.format(date=date_str)
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", REQUEST_HEADERS["User-Agent"])
        req.add_header("Accept-Encoding", "identity")
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        logger.error("Download failed for %s: %s", url, e)
        return pd.DataFrame()
    
    # Try gzip decompression, fall back to raw bytes
    try:
        try:
            decompressed = gzip.decompress(raw_bytes)
        except (gzip.BadGzipFile, OSError):
            decompressed = raw_bytes
        records = json.loads(decompressed)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("JSON parsing failed for %s: %s", url, e)
        return pd.DataFrame()
    
    if not isinstance(records, list) or len(records) == 0:
        logger.warning("Empty data for year %d", year)
        return pd.DataFrame()
    
    # Extract columns from positional JSON arrays
    rows = []
    for rec in records:
        if not isinstance(rec, list) or len(rec) < 45:
            continue
        row = {col_name: rec[idx] for col_name, idx in COL_IDX.items()}
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["Year"] = year
    
    # Clean team names
    df["Team"] = df["Team"].astype(str).str.strip()
    
    # Ensure numeric columns
    numeric_cols = [c for c in df.columns if c not in ("Team", "Year")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    logger.info("Downloaded %d teams for year %d (snapshot: %s)", len(df), year, date_str)
    return df


def download_all_snapshots(
    years: list[int] | None = None,
    output_path: str = "data_files/barttorvik_extended.csv",
) -> pd.DataFrame:
    """Download BartTorvik Time Machine data for all available years.
    
    Args:
        years: Specific years to download, or None for all available.
        output_path: Output CSV path.
        
    Returns:
        Combined DataFrame.
    """
    years = years or sorted(SNAPSHOT_DATES.keys())
    frames = [download_barttorvik_snapshot(y) for y in years]
    frames = [f for f in frames if not f.empty]
    
    if not frames:
        logger.warning("No data downloaded")
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows (%d years) to %s", len(df), len(frames), output_path)
    return df
```

### Key Advantages
1. **Pre-tournament snapshots** — eliminates data leakage from tournament results
2. **13 stats vs our 2** — WAB, EliteSOS, QualO, QualD, QualBarthag, Barthag, SOS, etc.
3. **No Selenium** — uses plain `urllib` (standard library)
4. **Historical data back to 2010** — 15 years of training data
5. **Compressed JSON** — fast downloads, small payloads

### Data Leakage Warning
Our current BartTorvik download (`barttorvik.com/team-tables_each.php?csv=1`) pulls **season-end** data that includes tournament results. When training a model to predict tournament outcomes, this is data leakage — the model has access to information from the future. The Time Machine approach fixes this completely.

---

## 3. ESPN Injury Scraping (New Data Source)

Not currently used in either repo, but specified in sports-quant's injury-adjustment-spec:

```python
# espn_injury_scraper.py — Fetch injury reports

import logging
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InjuryStatus(Enum):
    OUT = "out"
    DOUBTFUL = "doubtful"
    QUESTIONABLE = "questionable"
    PROBABLE = "probable"
    HEALTHY = "healthy"


@dataclass(frozen=True)
class InjuryReport:
    player_name: str
    team: str
    status: InjuryStatus
    injury_description: str
    report_date: str
    source: str  # "espn", "manual", "llm"


def scrape_espn_injuries() -> list[InjuryReport]:
    """Scrape ESPN college basketball injury page.
    
    URL: https://www.espn.com/mens-college-basketball/injuries
    
    Returns:
        List of InjuryReport objects for all reported injuries.
    """
    url = "https://www.espn.com/mens-college-basketball/injuries"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("ESPN injury fetch failed: %s", e)
        return []
    
    soup = BeautifulSoup(resp.text, "html.parser")
    injuries = []
    
    # Parse injury tables — structure may change, adapt selectors as needed
    # ESPN typically has team sections with player rows
    team_sections = soup.find_all("div", class_="ResponsiveTable")
    
    for section in team_sections:
        team_header = section.find("div", class_="Table__Title")
        team_name = team_header.get_text(strip=True) if team_header else "Unknown"
        
        rows = section.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 3:
                player = cells[0].get_text(strip=True)
                status_text = cells[1].get_text(strip=True).lower()
                description = cells[2].get_text(strip=True)
                
                status = InjuryStatus.HEALTHY
                if "out" in status_text:
                    status = InjuryStatus.OUT
                elif "doubtful" in status_text:
                    status = InjuryStatus.DOUBTFUL
                elif "questionable" in status_text:
                    status = InjuryStatus.QUESTIONABLE
                elif "probable" in status_text:
                    status = InjuryStatus.PROBABLE
                
                if status != InjuryStatus.HEALTHY:
                    injuries.append(InjuryReport(
                        player_name=player,
                        team=team_name,
                        status=status,
                        injury_description=description,
                        report_date="",  # Parse from page if available
                        source="espn",
                    ))
    
    logger.info("Scraped %d injury reports from ESPN", len(injuries))
    return injuries
```

---

## 4. Sports Reference Player Stats (New Data Source)

For injury impact calculation — need player-level stats to quantify how much a player contributes:

```python
# sportsref_scraper.py — Player stats for injury impact scoring

import logging
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# URL pattern for team pages
SPORTSREF_URL = "https://www.sports-reference.com/cbb/schools/{slug}/{year}.html"

# Rate limit: ~20 requests/minute (3-second delays)
REQUEST_DELAY = 3.0

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_player_stats(team_slug: str, year: int) -> pd.DataFrame:
    """Scrape player stats from sports-reference.com for one team.
    
    Gets per-game and advanced stats for all players on a team roster.
    
    Args:
        team_slug: Sports-reference team slug (e.g. "north-carolina").
        year: Season year.
        
    Returns:
        DataFrame with player stats, or empty on error.
    """
    url = SPORTSREF_URL.format(slug=team_slug, year=year)
    
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return pd.DataFrame()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Parse Per Game stats table
    per_game_table = soup.find("table", {"id": "per_game"})
    if per_game_table is None:
        logger.warning("No per_game table at %s", url)
        return pd.DataFrame()
    
    try:
        per_game_df = pd.read_html(str(per_game_table))[0]
    except Exception as e:
        logger.error("Failed to parse per_game table: %s", e)
        return pd.DataFrame()
    
    # Parse Advanced stats table
    advanced_table = soup.find("table", {"id": "advanced"})
    if advanced_table is not None:
        try:
            advanced_df = pd.read_html(str(advanced_table))[0]
            # Merge on player name
            per_game_df = per_game_df.merge(
                advanced_df[["Player", "USG%", "ORtg", "DRtg", "BPM", "OBPM", "DBPM"]],
                on="Player", how="left",
            )
        except Exception:
            pass  # Advanced stats optional
    
    per_game_df["team_slug"] = team_slug
    per_game_df["year"] = year
    
    time.sleep(REQUEST_DELAY)  # Rate limiting
    return per_game_df
```

---

## 5. Data Quality: Pre-Tournament Snapshot Strategy

A critical insight from sports-quant: **all training data must use pre-tournament snapshots**.

### The Problem
KenPom and BartTorvik update their ratings after every game, including tournament games. If you train on season-end data:
- For a model predicting R64 outcomes, the training data includes information from tournament rounds that haven't happened yet
- The model learns from "future" information it won't have at prediction time
- This inflates training accuracy and deflates real-world performance

### The Solution
Use **day-before-R64 snapshots** for all team ratings:
- KenPom: Web Archive snapshots from Selection Sunday week
- BartTorvik: Time Machine daily snapshots
- This ensures the model only sees information available before the tournament begins

### Implementation Note
Our current BartTorvik download (`download_barttorvik.py`) should be replaced or supplemented with the Time Machine approach. The existing Selenium-based CSV download can remain as a fallback for current-season live data, but all historical training data should use Time Machine snapshots.

---

## 6. Comparison: Our Scrapers vs sports-quant Scrapers

| Aspect | Our KenPom | sports-quant KenPom | Our BartTorvik | sports-quant BartTorvik |
|--------|-----------|-------------------|---------------|----------------------|
| Method | Selenium | requests + BS4 | Selenium | urllib |
| Source | Live page | Web Archive | CSV download | Time Machine JSON |
| Dependencies | Chrome, ChromeDriver | requests, bs4 | Chrome, ChromeDriver | None (stdlib) |
| Historical data | Current year only | 2017-2025 | Current year only | 2010-2025 |
| Stats captured | ~21 columns | ~21 columns | 2 (AdjOE, AdjDE) | 13 columns |
| Data leakage risk | Yes (live) | No (pre-tournament) | Yes (season-end) | No (pre-tournament) |
| Reliability | Medium (Selenium fragile) | High (static pages) | Medium | High |

---

## Priority Implementation Order

1. **BartTorvik Time Machine** — Immediate value: 13 stats, no Selenium, no data leakage
2. **KenPom Web Archive** — Replace Selenium dependency, get historical data
3. **ESPN Injuries** — New data dimension for prediction adjustments
4. **Sports Reference Player Stats** — Required for injury impact scoring
