"""
Ratings Page
Team efficiency ratings from EvanMiya and Haslametrics.
"""

import re
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Deployment environment detection
# Streamlit Community Cloud sets STREAMLIT_SHARING_MODE=1 at runtime.
# We also check for the common cloud home directory as a fallback.
# ---------------------------------------------------------------------------
IS_CLOUD = (
    os.environ.get("STREAMLIT_SHARING_MODE", "") == "1"
    or os.environ.get("IS_STREAMLIT_CLOUD", "").lower() == "true"
    or os.path.exists("/mount/src")
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ratings - Bracket Oracle",
    page_icon="📊",
    layout="wide",
)

logo_path = Path("data_files/logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=250)

st.title("📊 Ratings")
st.markdown("*Team efficiency ratings from EvanMiya, Haslametrics, and NCAA NET*")
CACHE_PATH = Path("data_files/evanmiya_ratings.csv")

# ---------------------------------------------------------------------------
# Expected column order (matches the screenshot)
# ---------------------------------------------------------------------------

EXPECTED_COLS = [
    "Relative Ranking",
    "Team",
    "O-Rate",
    "D-Rate",
    "Relative Rating",
    "Opponent Adjust",
    "Pace Adjust",
    "Off Rank",
    "Def Rank",
    "True Tempo",
    "Tempo Rank",
    "Injury Rank",
    "Home Rank",
    "Roster Rank",
    "Kill Shots Per Game",
    "Kill Shots Conceded Per Game",
    "Kill Shots Margin Per Game",
    "Total Kill Shots",
    "Total Kill Shots Conceded",
    "D1 Wins",
    "D1 Losses",
]

# Normalise whatever header text the table returns → canonical name
_HEADER_ALIASES: dict[str, str] = {
    # Ranking variants
    "relative ranking": "Relative Ranking",
    "relative\nranking": "Relative Ranking",
    "rk": "Relative Ranking",
    "rank": "Relative Ranking",
    "#": "Relative Ranking",
    "ranking": "Relative Ranking",
    # Team
    "team": "Team",
    # Offense
    "o-rate": "O-Rate",
    "o rate": "O-Rate",
    "orate": "O-Rate",
    "offensive rating": "O-Rate",
    "off rating": "O-Rate",
    "ortg": "O-Rate",
    # Defense
    "d-rate": "D-Rate",
    "d rate": "D-Rate",
    "drate": "D-Rate",
    "defensive rating": "D-Rate",
    "def rating": "D-Rate",
    "drtg": "D-Rate",
    # Net
    "relative rating": "Relative Rating",
    "net rating": "Relative Rating",
    "net rtg": "Relative Rating",
    "net": "Relative Rating",
    "bpr": "Relative Rating",
    # Adjustments
    "opponent adjust": "Opponent Adjust",
    "opponent\nadjust": "Opponent Adjust",
    "opp adjust": "Opponent Adjust",
    "sos adjust": "Opponent Adjust",
    "pace adjust": "Pace Adjust",
    "pace\nadjust": "Pace Adjust",
    "tempo adjust": "Pace Adjust",
    # Ranks
    "off rank": "Off Rank",
    "off\nrank": "Off Rank",
    "offensive rank": "Off Rank",
    "def rank": "Def Rank",
    "def\nrank": "Def Rank",
    "defensive rank": "Def Rank",
    # Tempo
    "true tempo": "True Tempo",
    "true\ntempo": "True Tempo",
    "tempo": "True Tempo",
    "adj t": "True Tempo",
    "adjt": "True Tempo",
    "tempo rank": "Tempo Rank",
    "tempo\nrank": "Tempo Rank",
    "pace rank": "Tempo Rank",
    # Injury
    "injury rank": "Injury Rank",
    "inj rank": "Injury Rank",
    "injury\nrank": "Injury Rank",
    # Home / Roster
    "home rank": "Home Rank",
    "home\nrank": "Home Rank",
    "roster rank": "Roster Rank",
    "roster\nrank": "Roster Rank",
    # Kill shots
    "kill shots per game": "Kill Shots Per Game",
    "kill shots pg": "Kill Shots Per Game",
    "kill shots\nper game": "Kill Shots Per Game",
    "ks per game": "Kill Shots Per Game",
    "kill shots conceded per game": "Kill Shots Conceded Per Game",
    "kill shots conceded pg": "Kill Shots Conceded Per Game",
    "kill shots conceded\nper game": "Kill Shots Conceded Per Game",
    "ksc per game": "Kill Shots Conceded Per Game",
    "kill shots margin per game": "Kill Shots Margin Per Game",
    "kill shots margin\nper game": "Kill Shots Margin Per Game",
    "ks margin per game": "Kill Shots Margin Per Game",
    "ks margin pg": "Kill Shots Margin Per Game",
    "total kill shots": "Total Kill Shots",
    "total kill\nshots": "Total Kill Shots",
    "total ks": "Total Kill Shots",
    "total kill shots conceded": "Total Kill Shots Conceded",
    "total kill shots\nconceded": "Total Kill Shots Conceded",
    "total kill\nshots conceded": "Total Kill Shots Conceded",
    "total ksc": "Total Kill Shots Conceded",
    # Record
    "d1 wins": "D1 Wins",
    "d1\nwins": "D1 Wins",
    "d1wins": "D1 Wins",
    "wins": "D1 Wins",
    "d1 losses": "D1 Losses",
    "d1\nlosses": "D1 Losses",
    "d1losses": "D1 Losses",
    "losses": "D1 Losses",
}

# ---------------------------------------------------------------------------
# Selenium driver
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
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
# API-response parser (used by network-interception strategy)
# ---------------------------------------------------------------------------


def _parse_api_responses(captured: list) -> "pd.DataFrame | None":
    """Scan intercepted fetch/XHR responses for the largest team-data payload."""
    import json as _json

    best_df: "pd.DataFrame | None" = None
    best_n = 0

    for item in captured:
        body = item.get("b", "") or ""
        if len(body) < 500:
            continue
        try:
            data = _json.loads(body)
        except (ValueError, TypeError):
            continue

        candidates: list = []
        if isinstance(data, list):
            candidates.append(data)
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    candidates.append(v)

        for lst in candidates:
            if len(lst) <= best_n:
                continue
            if not lst or not isinstance(lst[0], dict):
                continue
            df = pd.DataFrame(lst)
            # Must have at least one numeric-looking column to be team data
            if df.select_dtypes(include="number").shape[1] == 0:
                # Try coercing — real team rows have many numeric fields
                numeric_count = sum(
                    pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5
                    for c in df.columns
                )
                if numeric_count < 3:
                    continue
            best_df = df
            best_n = len(df)

    return best_df if best_n > 50 else None


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------


def scrape_evanmiya_team_ratings(year: str = "2025-26", progress_cb=None) -> pd.DataFrame:
    """
    Load evanmiya.com/?team_ratings and extract the team ratings table.

    Parameters
    ----------
    year:
        Year string shown in the site's Year dropdown, e.g. "2025-26".
    progress_cb:
        Optional callable(message: str, pct: float).

    Returns
    -------
    pd.DataFrame with columns matching EXPECTED_COLS where available.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from bs4 import BeautifulSoup

    from bs4 import BeautifulSoup

    def _cb(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    import json as _json

    # JS injected before page load — intercepts every fetch/XHR call so we can
    # read the raw API payload (which may contain ALL teams before the UI filters).
    _INTERCEPTOR_JS = """
    window.__ev_caps = [];
    const _f = window.fetch;
    window.fetch = async (...a) => {
        const url = (typeof a[0]==='string') ? a[0] : (a[0]&&a[0].url)||'';
        const resp = await _f(...a);
        try { const t = await resp.clone().text(); window.__ev_caps.push({u:url,b:t,s:resp.status}); } catch(_){}
        return resp;
    };
    const _xo = XMLHttpRequest.prototype.open, _xs = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function(m,u){ this.__u=u; return _xo.apply(this,arguments); };
    XMLHttpRequest.prototype.send = function(){
        this.addEventListener('load', () => window.__ev_caps.push({u:this.__u||'',b:this.responseText,s:this.status}));
        return _xs.apply(this,arguments);
    };
    """

    _cb("Launching browser…", 0.05)
    driver = _build_driver()

    try:
        # Inject interceptor BEFORE navigation so no requests are missed
        try:
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": _INTERCEPTOR_JS})
        except Exception:
            pass  # CDP not available in all environments — scroll fallback still works

        _cb("Loading evanmiya.com/?team_ratings…", 0.10)
        driver.get("https://evanmiya.com/?team_ratings")

        _cb("Waiting for React app to load…", 0.20)
        time.sleep(10)

        _select_year(driver, year, _cb)

        # ------------------------------------------------------------------
        # Strategy 1: extract from intercepted API responses
        # ------------------------------------------------------------------
        _cb("Checking intercepted API responses…", 0.32)
        captured = driver.execute_script("return window.__ev_caps || []") or []
        api_df = _parse_api_responses(captured)
        if api_df is not None:
            _cb(f"Got {len(api_df)} teams from API payload…", 0.85)
            n = len(api_df.columns)
            api_df.columns = list(EXPECTED_COLS[:n]) + [f"Col{i}" for i in range(n - len(EXPECTED_COLS))]
            return _normalise(api_df)

        # ------------------------------------------------------------------
        # Strategy 2: scroll through virtualized table
        # ------------------------------------------------------------------
        _cb("API interception yielded <50 teams — falling back to table scroll…", 0.34)

        # Mark the page's inner scrollable container so we can target it.
        # EvanMiya renders a virtualized table — only visible rows exist in
        # the DOM at any time.  We must scroll incrementally and accumulate.
        # ------------------------------------------------------------------
        driver.execute_script("""
            const els = Array.from(document.querySelectorAll('*')).reverse();
            for (const el of els) {
                const s = window.getComputedStyle(el);
                if ((s.overflowY === 'auto' || s.overflowY === 'scroll')
                    && el.scrollHeight > el.clientHeight + 50
                    && el.getBoundingClientRect().height > 100) {
                    el.setAttribute('data-evanmiya-scroller', '1');
                    break;
                }
            }
        """)

        _cb("Collecting rows (scrolling through all teams)…", 0.30)

        # Accumulate rows: keyed by team name to deduplicate across snapshots
        seen: dict[str, list[str]] = {}  # team_key -> row cells
        headers: list[str] = []
        SCROLL_STEP  = 150   # px per step — small so virtualized rows aren't skipped
        MAX_STEPS    = 300   # safety cap (~300 × 150 px = 45 000 px max scroll)
        NO_NEW_LIMIT = 15   # steps with no new teams before giving up (be patient)

        def _parse_snapshot() -> tuple[list[str], list[list[str]]]:
            """Parse the current DOM snapshot; return (headers, data_rows)."""
            soup = BeautifulSoup(driver.page_source, "html.parser")

            hdrs: list[str] = []
            rows: list[list[str]] = []

            # --- headers: <th> or role="columnheader" ---
            ths = soup.find_all("th")
            if ths:
                hdrs = [th.get_text(strip=True) for th in ths]
            else:
                chs = soup.find_all(attrs={"role": "columnheader"})
                if chs:
                    hdrs = [c.get_text(strip=True) for c in chs]

            # --- rows: <tr><td> ---
            for tbl in soup.find_all("table"):
                for tr in tbl.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) >= 3:
                        rows.append([td.get_text(strip=True) for td in tds])
            if rows:
                return hdrs, rows

            # --- ARIA grid ---
            for row in soup.find_all(attrs={"role": "row"}):
                cells = row.find_all(attrs={"role": ["cell", "gridcell"]})
                if len(cells) >= 3:
                    rows.append([c.get_text(strip=True) for c in cells])
            return hdrs, rows

        def _scroll_down():
            """Scroll the inner table container, or the window as fallback."""
            driver.execute_script(f"""
                const el = document.querySelector('[data-evanmiya-scroller]');
                if (el) {{
                    el.scrollTop += {SCROLL_STEP};
                }} else {{
                    window.scrollBy(0, {SCROLL_STEP});
                }}
            """)

        def _current_scroll() -> int:
            return driver.execute_script("""
                const el = document.querySelector('[data-evanmiya-scroller]');
                return el ? el.scrollTop : window.scrollY;
            """)

        # Strip emojis/non-ASCII to get a canonical team name for deduplication.
        # EvanMiya renders the same team multiple times (with and without emoji
        # badges) and also renders the top player for each team as a separate row.
        # We only want the full team row (which has all 21 columns filled).
        def _canonical_key(s: str) -> str:
            return re.sub(r'[^\x00-\x7F]+', '', s).strip()

        seen_fill: dict[str, int] = {}  # key -> filled-cell count of stored row

        last_scroll = -1
        no_new_streak = 0

        for step in range(MAX_STEPS):
            hdrs, rows = _parse_snapshot()
            if hdrs and not headers:
                headers = hdrs

            new_count = 0
            for row in rows:
                # col[0] must be a valid integer ranking
                raw_rank = row[0].strip() if row else ""
                if not raw_rank or not raw_rank.isdigit():
                    continue

                # col[1] is the team name
                team_cell = row[1].strip() if len(row) > 1 else ""
                key = _canonical_key(team_cell)
                if not key:
                    continue

                # Only accept full team rows — partial/player rows have ~5 cells
                # filled while complete team rows have all 21 columns filled.
                filled = sum(1 for c in row if c.strip())
                if filled < 15:
                    continue

                if key not in seen or filled > seen_fill[key]:
                    if key not in seen:
                        new_count += 1
                    seen[key] = row
                    seen_fill[key] = filled

            pct = min(0.30 + (step / MAX_STEPS) * 0.50, 0.79)
            _cb(f"Scrolling… {len(seen)} teams collected", pct)

            cur_scroll = _current_scroll()
            if cur_scroll == last_scroll and step > 0:
                break   # Hit the bottom
            last_scroll = cur_scroll

            if new_count == 0:
                no_new_streak += 1
                if no_new_streak >= NO_NEW_LIMIT:
                    break  # No new rows for several steps — assume complete
            else:
                no_new_streak = 0

            _scroll_down()
            time.sleep(0.4)

    finally:
        driver.quit()

    if not seen:
        raise RuntimeError(
            "Could not extract any tabular data from evanmiya.com/?team_ratings. "
            "The page may have changed structure, or the site may require a login."
        )

    _cb(f"Building DataFrame from {len(seen)} teams…", 0.82)
    df = _to_df(headers, list(seen.values()))

    # Always assign canonical column names positionally — the site's internal
    # header text (OBPR, DBPR, Net Rate, Projected 3PT%, etc.) does not match
    # the visible UI labels. Column ORDER is stable, so positional mapping is
    # the only reliable approach.
    n = len(df.columns)
    df.columns = list(EXPECTED_COLS[:n]) + [f"Col{i}" for i in range(n - len(EXPECTED_COLS))]

    _cb("Normalising columns…", 0.88)
    df = _normalise(df)

    _cb("Done!", 1.0)
    return df


# ---------------------------------------------------------------------------
# Year selector
# ---------------------------------------------------------------------------


def _select_year(driver, year: str, _cb):
    """Try to click the Year dropdown and select the requested year."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException

    try:
        # Generic: find a <select> whose current value doesn't match and change it
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel in selects:
            opts = sel.find_elements(By.TAG_NAME, "option")
            for opt in opts:
                if year in opt.text:
                    if opt.get_attribute("selected") is None:
                        _cb(f"Selecting year {year}…", 0.35)
                        from selenium.webdriver.support.ui import Select
                        Select(sel).select_by_visible_text(opt.text)
                        time.sleep(4)  # wait for data to reload
                    return
    except Exception:
        pass  # Year selection is best-effort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_df(headers: list[str], rows: list[list[str]]) -> pd.DataFrame:
    max_cols = max(len(headers) if headers else 0, *(len(r) for r in rows))
    if not headers:
        headers = [f"Col{i}" for i in range(max_cols)]
    while len(headers) < max_cols:
        headers.append(f"Col{len(headers)}")
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=headers[:max_cols])


def _is_numeric(text: str) -> bool:
    try:
        float(text.replace("+", "").replace(",", "").replace("−", "-").replace("–", "-"))
        return True
    except ValueError:
        return False


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using _HEADER_ALIASES, clean numerics, drop junk rows."""

    # Rename columns — normalise whitespace/newlines before alias lookup
    new_cols = []
    for col in df.columns:
        key = str(col).strip().lower()
        key = key.rstrip(" ↓↑")            # strip sort arrows
        key = " ".join(key.split())       # collapse internal whitespace/newlines
        new_cols.append(_HEADER_ALIASES.get(key, col.strip()))
    df.columns = new_cols

    # Deduplicate column names — duplicate names cause df[col] to return a
    # DataFrame instead of a Series, breaking every subsequent .str accessor.
    seen_cols: dict[str, int] = {}
    deduped: list[str] = []
    for c in df.columns:
        if c in seen_cols:
            seen_cols[c] += 1
            deduped.append(f"{c}.{seen_cols[c]}")
        else:
            seen_cols[c] = 0
            deduped.append(c)
    df.columns = deduped

    # Drop fully empty rows / columns
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    df = df.dropna(axis=1, how="all")

    # Coerce numeric columns
    numeric_cols = [
        "Relative Ranking",
        "O-Rate", "D-Rate", "Relative Rating",
        "Opponent Adjust", "Pace Adjust",
        "Off Rank", "Def Rank",
        "True Tempo", "Tempo Rank", "Injury Rank",
        "Home Rank", "Roster Rank",
        "Kill Shots Per Game", "Kill Shots Conceded Per Game", "Kill Shots Margin Per Game",
        "Total Kill Shots", "Total Kill Shots Conceded",
        "D1 Wins", "D1 Losses",
    ]
    for col in numeric_cols:
        if col in df.columns:
            series = df[col]
            # Guard: if somehow still a DataFrame, take the first column
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            df[col] = (
                series
                .astype(str)
                .str.replace("+", "", regex=False)
                .str.replace("−", "-", regex=False)
                .str.replace("–", "-", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

    # Drop header-repeat rows (where "Team" column contains "Team")
    if "Team" in df.columns:
        team_col = df["Team"]
        if isinstance(team_col, pd.DataFrame):
            team_col = team_col.iloc[:, 0]
        df = df[~team_col.astype(str).str.strip().str.lower().eq("team")]
        team_col = df["Team"]
        if isinstance(team_col, pd.DataFrame):
            team_col = team_col.iloc[:, 0]
        df = df[team_col.notna() & (team_col.astype(str).str.strip() != "")]

    # Strip emojis/badge icons from Team names (🏀🤕📉🔒🥶💥🔥 etc.)
    if "Team" in df.columns:
        team_col = df["Team"]
        if isinstance(team_col, pd.DataFrame):
            team_col = team_col.iloc[:, 0]
        df["Team"] = team_col.astype(str).apply(
            lambda s: re.sub(r'[^\x00-\x7F]+', '', s).strip()
        )

    # Sort
    if "Relative Ranking" in df.columns and df["Relative Ranking"].notna().any():
        df = df.sort_values("Relative Ranking").reset_index(drop=True)
    elif "Relative Rating" in df.columns and df["Relative Rating"].notna().any():
        df = df.sort_values("Relative Rating", ascending=False).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Controls")

year_choice = st.sidebar.selectbox(
    "Year",
    ["2025-26", "2024-25", "2023-24", "2022-23"],
    index=0,
    help="Season year (passed to EvanMiya's Year filter)",
)

include_injuries = st.sidebar.selectbox(
    "Include Current Injuries?",
    ["No", "Yes"],
    index=0,
)

conf_filter = st.sidebar.text_input(
    "Filter by Conference",
    placeholder="e.g. B10, ACC, SEC",
)

min_rating = st.sidebar.slider(
    "Minimum Relative Rating",
    min_value=-20.0,
    max_value=40.0,
    value=-20.0,
    step=0.5,
)

show_top_n = st.sidebar.slider(
    "Show Top N Teams",
    min_value=10,
    max_value=400,
    value=400,
    step=10,
)

if IS_CLOUD:
    refresh = False
    clear_cache = False
else:
    refresh = st.sidebar.button("🔄 Fetch Fresh Data from EvanMiya", type="primary")
    clear_cache = st.sidebar.button("🗑️ Delete Cache & Re-fetch", help="Deletes the local CSV and forces a full re-scrape")
    if clear_cache:
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
            st.sidebar.success("Cache deleted — click Fetch Fresh Data to re-scrape.")
        refresh = True

# ---------------------------------------------------------------------------
# Scrape or load cached
# ---------------------------------------------------------------------------

df: pd.DataFrame | None = None

if refresh:
    progress_bar = st.progress(0.0, text="Starting…")
    status_txt = st.empty()

    def _progress(msg: str, pct: float):
        progress_bar.progress(min(pct, 1.0), text=msg)
        status_txt.text(msg)

    try:
        with st.spinner("Scraping evanmiya.com — takes ~20–30 seconds…"):
            df = scrape_evanmiya_team_ratings(year=year_choice, progress_cb=_progress)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        progress_bar.empty()
        status_txt.empty()
        st.success(f"✅ Fetched {len(df)} teams. Cached to `{CACHE_PATH}`.")
    except Exception as exc:
        progress_bar.empty()
        status_txt.empty()
        st.error(f"❌ Scrape failed: {exc}")
        st.info(
            "evanmiya.com may be temporarily unreachable, or its page structure may "
            "have changed. Try again later, or place a manually exported CSV at "
            "`data_files/evanmiya_ratings.csv`."
        )

if df is None:
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH)
        mtime = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(CACHE_PATH.stat().st_mtime)
        )
        st.caption(
            f"Showing cached data (last updated {mtime}). "
            "Click **Fetch Fresh Data** in the sidebar to update."
        )
    else:
        st.info(
            "No cached data found."
            + (" Click **🔄 Fetch Fresh Data from EvanMiya** in the sidebar, or upload an exported CSV below." if not IS_CLOUD else "")
        )

if not IS_CLOUD:
    # -------------------------------------------------------------------------
    # CSV upload fallback (works even when scraping is blocked at 50 teams)
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Or upload a CSV exported from EvanMiya:**")
    uploaded = st.sidebar.file_uploader(
        "Upload evanmiya_ratings.csv",
        type="csv",
        help="Export from EvanMiya → paste into a CSV with the standard column order, then upload here.",
    )
    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
            # If columns look generic, assign positionally
            if not any(str(c) in EXPECTED_COLS for c in up_df.columns):
                n = len(up_df.columns)
                up_df.columns = list(EXPECTED_COLS[:n]) + [f"Col{i}" for i in range(n - len(EXPECTED_COLS))]
            up_df = _normalise(up_df)
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            up_df.to_csv(CACHE_PATH, index=False)
            df = up_df
            st.sidebar.success(f"✅ Loaded {len(df)} teams from upload.")
        except Exception as exc:
            st.sidebar.error(f"Upload failed: {exc}")

if df is None:
    st.stop()

if df is None or df.empty:
    st.warning("No data available.")
    st.stop()

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

filtered = df.copy()

if conf_filter.strip() and "Conf" in filtered.columns:
    filtered = filtered[
        filtered["Conf"].astype(str).str.contains(
            conf_filter.strip(), case=False, na=False
        )
    ]

if "Relative Rating" in filtered.columns and filtered["Relative Rating"].notna().any():
    filtered = filtered[filtered["Relative Rating"].fillna(-999) >= min_rating]

filtered = filtered.head(show_top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_evan, tab_hasla, tab_net = st.tabs(["🧠 EvanMiya", "📐 Haslametrics", "🏆 NCAA NET"])

# ===========================================================================
# TAB 1 — EvanMiya
# ===========================================================================
with tab_evan:
    # ---------------------------------------------------------------------------
    # Summary metrics row
    # ---------------------------------------------------------------------------

    st.header("📊 Summary")

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Teams shown", len(filtered))

    if "Relative Rating" in df.columns and df["Relative Rating"].notna().any():
        top_idx = df["Relative Rating"].idxmax()
        with metric_cols[1]:
            top_name = df.loc[top_idx, "Team"] if "Team" in df.columns else "—"
            st.metric("Highest Relative Rating", f"{df.loc[top_idx, 'Relative Rating']:.1f}", delta=top_name)

    if "O-Rate" in df.columns and df["O-Rate"].notna().any():
        best_off_idx = df["O-Rate"].idxmax()
        with metric_cols[2]:
            best_off = df.loc[best_off_idx, "Team"] if "Team" in df.columns else "—"
            st.metric("Best Offense (O-Rate)", f"{df.loc[best_off_idx, 'O-Rate']:.1f}", delta=best_off)

    if "D-Rate" in df.columns and df["D-Rate"].notna().any():
        best_def_idx = df["D-Rate"].idxmax()
        with metric_cols[3]:
            best_def = df.loc[best_def_idx, "Team"] if "Team" in df.columns else "—"
            st.metric("Best Defense (D-Rate)", f"{df.loc[best_def_idx, 'D-Rate']:.1f}", delta=best_def)

    # ---------------------------------------------------------------------------
    # Main table
    # ---------------------------------------------------------------------------

    st.header("📋 Team Ratings")

    # Column display order: priority cols first, then any extras
    disp_order = [c for c in EXPECTED_COLS if c in filtered.columns]
    extra = [c for c in filtered.columns if c not in disp_order]
    disp_order += extra

    disp = filtered[disp_order].copy()

    # Format numeric columns
    for col in ["O-Rate", "D-Rate", "Relative Rating", "Opponent Adjust", "Pace Adjust", "True Tempo",
                "Kill Shots Per Game", "Kill Shots Conceded Per Game", "Kill Shots Margin Per Game"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    for col in ["Relative Ranking", "Off Rank", "Def Rank", "Tempo Rank", "Injury Rank",
                "Home Rank", "Roster Rank",
                "Total Kill Shots", "Total Kill Shots Conceded"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

    st.dataframe(disp, hide_index=True, width="stretch")

    # ---------------------------------------------------------------------------
    # Scatter: O-Rate vs D-Rate  (matches EvanMiya's own layout instinct)
    # ---------------------------------------------------------------------------

    if "O-Rate" in filtered.columns and "D-Rate" in filtered.columns:
        st.header("📈 Offense vs Defense")

        scat = filtered[["Team", "O-Rate", "D-Rate"]].dropna()
        if "Conf" in filtered.columns:
            scat = scat.copy()
            scat["Conf"] = filtered.loc[scat.index, "Conf"].values

        if not scat.empty:
            color_col = "Conf" if "Conf" in scat.columns and scat["Conf"].notna().any() else None
            fig = px.scatter(
                scat,
                x="O-Rate",
                y="D-Rate",
                color=color_col,
                text="Team",
                title="O-Rate vs D-Rate  (both higher = better)",
                labels={
                    "O-Rate": "Offensive Rating (O-Rate) →",
                    "D-Rate": "Defensive Rating (D-Rate) →",
                },
            )
            fig.update_traces(textposition="top center", textfont_size=8, marker_size=7)

            # Median reference lines
            fig.add_vline(
                x=scat["O-Rate"].median(),
                line_dash="dash", line_color="lightgray", opacity=0.7,
                annotation_text="Median O-Rate",
            )
            fig.add_hline(
                y=scat["D-Rate"].median(),
                line_dash="dash", line_color="lightgray", opacity=0.7,
                annotation_text="Median D-Rate",
            )
            fig.update_layout(
                height=620,
                showlegend=(color_col is not None and scat[color_col].nunique() <= 25),
            )
            st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------------------------
    # Top 25 bar chart — Relative Rating
    # ---------------------------------------------------------------------------

    rating_col = next(
        (c for c in ["Relative Rating", "O-Rate"] if c in filtered.columns), None
    )
    if rating_col and "Team" in filtered.columns:
        st.header(f"🏆 Top 25 Teams — {rating_col}")

        top25 = (
            filtered[["Team", rating_col]].dropna().nlargest(25, rating_col)
        )
        if not top25.empty:
            bar_fig = px.bar(
                top25,
                x=rating_col,
                y="Team",
                orientation="h",
                color=rating_col,
                color_continuous_scale="RdYlGn",
                title=f"Top 25 by {rating_col}",
                labels={rating_col: rating_col, "Team": ""},
            )
            bar_fig.update_layout(
                yaxis={"autorange": "reversed"},
                height=680,
                coloraxis_showscale=False,
            )
            st.plotly_chart(bar_fig, width="stretch")

    # ---------------------------------------------------------------------------
    # Tempo distribution
    # ---------------------------------------------------------------------------

    if "True Tempo" in filtered.columns and filtered["True Tempo"].notna().any():
        st.header("⏱ Pace Distribution")

        hist_fig = px.histogram(
            filtered,
            x="True Tempo",
            nbins=30,
            title="Distribution of True Tempo (possessions per 40 min)",
            labels={"True Tempo": "True Tempo", "count": "Teams"},
            color_discrete_sequence=["#636EFA"],
        )
        hist_fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(hist_fig, width="stretch")

    # ---------------------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------------------

    st.markdown("---")
    st.caption(
        "Data scraped from [evanmiya.com/?team_ratings](https://evanmiya.com/?team_ratings). "
        "**O-Rate** = Offensive Rating, **D-Rate** = Defensive Rating, "
        "**Relative Rating** = net composite (O-Rate + D-Rate). "
        "Click **Fetch Fresh Data** in the sidebar to pull the latest numbers."
    )

# ===========================================================================
# TAB 2 — Haslametrics  (local only)
# ===========================================================================
with tab_hasla:
    import sys as _sys
    _sys.path.insert(0, ".")
    import subprocess as _subprocess
    from pathlib import Path as _Path

    HASLA_CACHE = _Path("data_files/haslametrics_ratings.csv")
    HASLA_CANONICAL = _Path("data_files/haslametrics_canonical.csv")

    st.markdown("*Offense and defense efficiency ratings from [haslametrics.com/ratings.php](https://haslametrics.com/ratings.php)*")

    if st.button("🔄 Refresh Haslametrics Data", type="primary", key="btn_hasla_refresh"):
        with st.spinner("Downloading from haslametrics.com… (takes ~30 seconds)"):
            try:
                result = _subprocess.run(
                    [_sys.executable, "download_haslametrics.py"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    st.success("✅ Haslametrics data refreshed.")
                    st.caption(result.stdout[-800:] if result.stdout else "")
                else:
                    st.error(f"Download failed:\n{result.stderr[-800:]}")
            except Exception as _e:
                st.error(f"Error: {_e}")

    hasla_df = None
    _cache_to_use = HASLA_CANONICAL if HASLA_CANONICAL.exists() else HASLA_CACHE
    if _cache_to_use.exists():
        hasla_df = pd.read_csv(_cache_to_use)
        hasla_df = hasla_df[hasla_df["O_Eff"].notna()].copy()
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(_cache_to_use.stat().st_mtime))
        st.caption(f"Last updated: {mtime} · {len(hasla_df)} teams · source: [haslametrics.com](https://haslametrics.com/ratings.php)")
    else:
        st.info("No Haslametrics data cached. Click **Refresh** above to download.")

    if hasla_df is not None and not hasla_df.empty:
        # -----------------------------------------------------------------------
        # Filter sidebar (Haslametrics-specific)
        # -----------------------------------------------------------------------
        hasla_min_o = st.sidebar.slider(
            "Hasla: Min Offensive Efficiency",
            min_value=float(int(hasla_df["O_Eff"].min() - 1)),
            max_value=float(int(hasla_df["O_Eff"].max() + 1)),
            value=float(int(hasla_df["O_Eff"].min() - 1)),
            step=1.0,
            key="hasla_min_o",
        )
        hasla_max_d = st.sidebar.slider(
            "Hasla: Max Defensive Efficiency",
            min_value=float(int(hasla_df["D_Eff"].min() - 1)),
            max_value=float(int(hasla_df["D_Eff"].max() + 1)),
            value=float(int(hasla_df["D_Eff"].max() + 1)),
            step=1.0,
            key="hasla_max_d",
        )

        filtered_h = hasla_df[
            (hasla_df["O_Eff"] >= hasla_min_o) & (hasla_df["D_Eff"] <= hasla_max_d)
        ].copy()

        # -----------------------------------------------------------------------
        # Summary metrics
        # -----------------------------------------------------------------------
        st.header("📊 Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Teams shown", len(filtered_h))
        best_net_idx = hasla_df["hasla_net_eff"].idxmax() if "hasla_net_eff" in hasla_df.columns else None
        if best_net_idx is not None:
            _nm = hasla_df.loc[best_net_idx, "canonical_team"] if "canonical_team" in hasla_df.columns else hasla_df.loc[best_net_idx, "Team"]
            with c2:
                st.metric("Best Net Efficiency", f"{hasla_df.loc[best_net_idx,'hasla_net_eff']:.1f}", delta=_nm)
        best_o_idx = hasla_df["O_Eff"].idxmax()
        with c3:
            _nm = hasla_df.loc[best_o_idx, "canonical_team"] if "canonical_team" in hasla_df.columns else hasla_df.loc[best_o_idx, "Team"]
            st.metric("Best Offense (O_Eff)", f"{hasla_df.loc[best_o_idx,'O_Eff']:.1f}", delta=_nm)
        best_d_idx = hasla_df["D_Eff"].idxmin()
        with c4:
            _nm = hasla_df.loc[best_d_idx, "canonical_team"] if "canonical_team" in hasla_df.columns else hasla_df.loc[best_d_idx, "Team"]
            st.metric("Best Defense (D_Eff)", f"{hasla_df.loc[best_d_idx,'D_Eff']:.1f}", delta=_nm)

        # -----------------------------------------------------------------------
        # Main table
        # -----------------------------------------------------------------------
        st.header("📋 Team Ratings")

        _name_col = "canonical_team" if "canonical_team" in filtered_h.columns else "Team"
        disp_h_cols = [_name_col, "Rk", "O_Eff", "D_Eff", "hasla_net_eff",
                       "O_AP%", "D_AP%", "O_FG%", "D_FG%", "O_3P%", "D_3P%",
                       "O_FTAR", "D_FTAR", "O_FT%", "D_FT%"]
        disp_h_cols = [c for c in disp_h_cols if c in filtered_h.columns]
        disp_h = filtered_h[disp_h_cols].copy()
        disp_h = disp_h.rename(columns={_name_col: "Team", "hasla_net_eff": "Net Eff"})
        disp_h = disp_h.sort_values("Rk").reset_index(drop=True)

        for col in ["O_Eff", "D_Eff", "Net Eff", "O_AP%", "D_AP%",
                    "O_FG%", "D_FG%", "O_3P%", "D_3P%", "O_FT%", "D_FT%",
                    "O_FTAR", "D_FTAR"]:
            if col in disp_h.columns:
                disp_h[col] = disp_h[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")

        if "Rk" in disp_h.columns:
            disp_h["Rk"] = disp_h["Rk"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

        st.dataframe(disp_h, hide_index=True, width="stretch")

        # -----------------------------------------------------------------------
        # Scatter: Offense vs Defense
        # -----------------------------------------------------------------------
        if "O_Eff" in filtered_h.columns and "D_Eff" in filtered_h.columns:
            st.header("📈 Offense vs Defense Efficiency")
            _nc = "canonical_team" if "canonical_team" in filtered_h.columns else "Team"
            scat_h = filtered_h[[_nc, "O_Eff", "D_Eff"]].dropna().rename(columns={_nc: "Team"})
            if not scat_h.empty:
                fig_h = px.scatter(
                    scat_h,
                    x="O_Eff",
                    y="D_Eff",
                    text="Team",
                    title="Haslametrics: Offensive Efficiency vs Defensive Efficiency",
                    labels={
                        "O_Eff": "Offensive Efficiency →",
                        "D_Eff": "Defensive Efficiency (lower = better) →",
                    },
                    color_discrete_sequence=["#EF553B"],
                )
                fig_h.update_traces(textposition="top center", textfont_size=8, marker_size=7)
                fig_h.add_vline(x=scat_h["O_Eff"].median(), line_dash="dash", line_color="lightgray", opacity=0.7)
                fig_h.add_hline(y=scat_h["D_Eff"].median(), line_dash="dash", line_color="lightgray", opacity=0.7)
                fig_h.update_layout(height=620)
                st.plotly_chart(fig_h, width="stretch")

        # -----------------------------------------------------------------------
        # Top 25 — Net Efficiency
        # -----------------------------------------------------------------------
        _net_col = "hasla_net_eff"
        if _net_col in hasla_df.columns:
            st.header("🏆 Top 25 Teams — Net Efficiency (O_Eff − D_Eff)")
            _nc = "canonical_team" if "canonical_team" in hasla_df.columns else "Team"
            top25_h = hasla_df[[_nc, _net_col]].dropna().nlargest(25, _net_col).rename(columns={_nc: "Team", _net_col: "Net Eff"})
            if not top25_h.empty:
                bar_h = px.bar(
                    top25_h,
                    x="Net Eff",
                    y="Team",
                    orientation="h",
                    color="Net Eff",
                    color_continuous_scale="RdYlGn",
                    title="Top 25 by Net Efficiency",
                    labels={"Net Eff": "Net Efficiency", "Team": ""},
                )
                bar_h.update_layout(yaxis={"autorange": "reversed"}, height=680, coloraxis_showscale=False)
                st.plotly_chart(bar_h, width="stretch")

        st.markdown("---")
        st.caption(
            "Data from [haslametrics.com/ratings.php](https://haslametrics.com/ratings.php). "
            "**O_Eff** = Offensive Efficiency, **D_Eff** = Defensive Efficiency, "
            "**Net Eff** = O_Eff − D_Eff. Click **Refresh** to pull the latest numbers."
        )

# ===========================================================================
# TAB 3 — NCAA NET Rankings  (local only)
# ===========================================================================
with tab_net:
    import sys as _sys2
    _sys2.path.insert(0, ".")
    import subprocess as _subprocess2
    from pathlib import Path as _Path2

    NET_CACHE = _Path2("data_files/net_rankings.csv")

    st.markdown(
        "*Official NCAA NET rankings from "
        "[ncaa.com](https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings)*"
    )

    if st.button("🔄 Refresh NCAA NET Rankings", type="primary", key="btn_net_refresh"):
        with st.spinner("Downloading from ncaa.com… (takes ~5 seconds)"):
            try:
                result = _subprocess2.run(
                    [_sys2.executable, "download_net_rankings.py"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    st.success("✅ NCAA NET Rankings refreshed.")
                else:
                    st.error(f"Download failed:\n{result.stderr[-800:]}")
            except Exception as _e:
                st.error(f"Error: {_e}")

    net_df = None
    if NET_CACHE.exists():
        net_df = pd.read_csv(NET_CACHE)
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(NET_CACHE.stat().st_mtime))
        st.caption(
            f"Last updated: {mtime} · {len(net_df)} teams · "
            f"source: [ncaa.com](https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings)"
        )
    else:
        st.info("No NET Rankings cached. Click **Refresh** above to download.")

    if net_df is not None and not net_df.empty:
        # -----------------------------------------------------------------------
        # Sidebar filters
        # -----------------------------------------------------------------------
        net_conf_filter = st.sidebar.text_input(
            "NET: Filter by Conference",
            placeholder="e.g. SEC, ACC, Big Ten",
            key="net_conf",
        )
        net_top_n = st.sidebar.slider(
            "NET: Show Top N Teams",
            min_value=10,
            max_value=len(net_df),
            value=min(100, len(net_df)),
            step=10,
            key="net_top_n",
        )

        # -----------------------------------------------------------------------
        # Summary metrics
        # -----------------------------------------------------------------------
        st.header("📊 Summary")
        c1, c2, c3, c4 = st.columns(4)

        _name_col_net = "canonical_team" if "canonical_team" in net_df.columns else "Team"
        _top1 = net_df.iloc[0]

        with c1:
            st.metric("Total Teams", len(net_df))
        with c2:
            st.metric("#1 NET", _top1.get(_name_col_net, _top1.get("Team", "—")), delta=str(_top1.get("Record", "")))

        if "Quad1" in net_df.columns:
            _best_q1_idx = net_df["Quad1"].apply(
                lambda x: int(x.split("-")[0]) if isinstance(x, str) and "-" in x else 0
            ).idxmax()
            _best_q1 = net_df.loc[_best_q1_idx]
            with c3:
                st.metric(
                    "Most Quad 1 Wins",
                    _best_q1.get(_name_col_net, _best_q1.get("Team", "—")),
                    delta=f"Q1: {_best_q1.get('Quad1', '')}",
                )

        with c4:
            _power = net_df[net_df["Conf"].isin(["ACC", "Big Ten", "Big 12", "SEC", "Big East"])] if "Conf" in net_df.columns else net_df
            st.metric("Power Conference Teams", len(_power))

        # -----------------------------------------------------------------------
        # Apply filters
        # -----------------------------------------------------------------------
        filtered_net = net_df.copy()
        if net_conf_filter.strip() and "Conf" in filtered_net.columns:
            filtered_net = filtered_net[
                filtered_net["Conf"].astype(str).str.contains(net_conf_filter.strip(), case=False, na=False)
            ]
        filtered_net = filtered_net.head(net_top_n).reset_index(drop=True)

        # -----------------------------------------------------------------------
        # Main table
        # -----------------------------------------------------------------------
        st.header("📋 NET Rankings Table")

        _name_col = "canonical_team" if "canonical_team" in filtered_net.columns else "Team"
        display_cols = [c for c in [
            "NET_Rank", _name_col, "Record", "Conf",
            "Road", "Neutral", "Home", "Non_DivI",
            "Quad1", "Quad2", "Quad3", "Quad4", "NET_Prev",
        ] if c in filtered_net.columns]

        disp_net = filtered_net[display_cols].copy()
        disp_net = disp_net.rename(columns={
            _name_col: "Team",
            "NET_Rank": "NET Rank",
            "NET_Prev": "Prev Rank",
            "Non_DivI": "Non-Div I",
        })
        for col in ["NET Rank", "Prev Rank"]:
            if col in disp_net.columns:
                disp_net[col] = disp_net[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

        st.dataframe(disp_net, hide_index=True, width="stretch")

        # -----------------------------------------------------------------------
        # Top 25 bar chart
        # -----------------------------------------------------------------------
        if "NET_Rank" in net_df.columns:
            st.header("🏆 Top 25 Teams — NET Ranking")
            _top25_net = net_df.nsmallest(25, "NET_Rank").copy()
            _top25_net["Label"] = _top25_net[_name_col_net]

            bar_net = px.bar(
                _top25_net,
                x="NET_Rank",
                y="Label",
                orientation="h",
                color="NET_Rank",
                color_continuous_scale="RdYlGn_r",
                hover_data=[c for c in ["Record", "Conf", "Quad1"] if c in _top25_net.columns],
                title="Top 25 by NET Rank (lower is better)",
                labels={"NET_Rank": "NET Rank", "Label": ""},
            )
            bar_net.update_layout(
                yaxis={"autorange": "reversed"},
                height=680,
                coloraxis_showscale=False,
            )
            st.plotly_chart(bar_net, width="stretch")

        # -----------------------------------------------------------------------
        # Quad 1 record scatter — Top 50
        # -----------------------------------------------------------------------
        if "Quad1" in net_df.columns:
            st.header("📊 Quad 1 Record vs NET Rank — Top 50 Teams")
            _q_df = net_df.nsmallest(50, "NET_Rank").copy()
            _q_df["Q1 Wins"] = _q_df["Quad1"].apply(
                lambda x: int(x.split("-")[0]) if isinstance(x, str) and "-" in x else 0
            )
            _q_df["Q1 Losses"] = _q_df["Quad1"].apply(
                lambda x: int(x.split("-")[1]) if isinstance(x, str) and "-" in x else 0
            )
            _q_df["TeamLabel"] = _q_df[_name_col_net]

            fig_q = px.scatter(
                _q_df,
                x="Q1 Wins",
                y="NET_Rank",
                text="TeamLabel",
                color="Q1 Losses",
                color_continuous_scale="RdYlGn_r",
                title="Quad 1 Wins vs NET Rank (Top 50 — lower rank = better)",
                labels={"NET_Rank": "NET Rank", "Q1 Wins": "Quad 1 Wins", "Q1 Losses": "Q1 Losses"},
            )
            fig_q.update_traces(textposition="top center", textfont_size=8)
            fig_q.update_yaxes(autorange="reversed")
            fig_q.update_layout(height=600)
            st.plotly_chart(fig_q, width="stretch")

        # -----------------------------------------------------------------------
        # Conference breakdown
        # -----------------------------------------------------------------------
        if "Conf" in net_df.columns:
            st.header("🏀 Conference Breakdown")
            conf_avg = (
                net_df.groupby("Conf")["NET_Rank"]
                .agg(["mean", "count", "min"])
                .rename(columns={"mean": "Avg NET Rank", "count": "Teams", "min": "Best Rank"})
                .sort_values("Avg NET Rank")
                .reset_index()
            )
            conf_avg["Avg NET Rank"] = conf_avg["Avg NET Rank"].round(1)
            conf_fig = px.bar(
                conf_avg.head(20),
                x="Avg NET Rank",
                y="Conf",
                orientation="h",
                color="Avg NET Rank",
                color_continuous_scale="RdYlGn_r",
                hover_data=["Teams", "Best Rank"],
                title="Top 20 Conferences by Average NET Rank (lower = better)",
            )
            conf_fig.update_layout(yaxis={"autorange": "reversed"}, height=560, coloraxis_showscale=False)
            st.plotly_chart(conf_fig, width="stretch")

        st.markdown("---")
        st.caption(
            "Data from [ncaa.com NET Rankings](https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings). "
            "**Quad 1**: away 1–30, neutral 1–50, home 1–75. "
            "**Quad 2**: away 31–75, neutral 51–100, home 76–135. "
            "Click **Refresh** to pull the latest rankings."
        )
