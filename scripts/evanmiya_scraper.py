"""
evanmiya.com Team Ratings Scraper
==================================
Requirements:
    pip install playwright pandas
    playwright install chromium

Usage:
    python evanmiya_scraper.py

Output:
    evanmiya_team_ratings.csv
"""

import pandas as pd
from playwright.sync_api import sync_playwright

TARGET_URL = "https://evanmiya.com/?team_ratings"
OUTPUT_CSV = "evanmiya_team_ratings.csv"


def dismiss_popup(page):
    strategies = [
        lambda: page.mouse.click(50, 50),
        lambda: page.keyboard.press("Escape"),
        lambda: _click_visible(page, [
            "button:has-text('Close')", "button:has-text('×')",
            "button:has-text('✕')",    "button:has-text('Done')",
            "button:has-text('Cancel')","button:has-text('Skip')",
            "[aria-label='Close']",    "[aria-label='close']",
            ".modal-close", ".close", ".dismiss",
        ]),
        lambda: _click_visible(page, [
            ".modal-backdrop", ".overlay", ".backdrop",
            "[class*='overlay']", "[class*='backdrop']", "[class*='modal']",
        ]),
    ]
    for fn in strategies:
        try:
            fn()
            page.wait_for_timeout(800)
            text = page.evaluate("() => document.body.innerText")
            if "Type or select a team" not in text:
                print("  ✓ Popup dismissed.")
                return
        except Exception:
            pass


def _click_visible(page, selectors):
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=800):
                el.click(timeout=800)
                return
        except Exception:
            continue


def extract_div_table(page) -> list:
    """
    Extract data from a div-based table by finding elements that contain
    team names and their sibling data cells.
    Uses innerText of the full rendered page, parsed row by row.
    """
    return page.evaluate("""
        () => {
            // ── Approach 1: find a container whose direct children look like rows ──
            // Walk every div, find one that has 50+ children (likely the table body)
            const allDivs = Array.from(document.querySelectorAll('div'));

            let bestContainer = null;
            let bestCount = 0;
            for (const div of allDivs) {
                const children = div.children;
                if (children.length > bestCount && children.length > 20) {
                    // Heuristic: rows should all have similar child counts
                    const childCounts = Array.from(children).map(c => c.children.length);
                    const mode = childCounts.sort((a,b) => a-b)[Math.floor(childCounts.length/2)];
                    if (mode > 3) {
                        bestCount = children.length;
                        bestContainer = div;
                    }
                }
            }

            if (!bestContainer) return { error: 'No container found', rows: [] };

            // Get header row — look for a sibling or parent that has header text
            const rows = [];
            const children = Array.from(bestContainer.children);

            // Grab text content of each child row's cells
            children.forEach(row => {
                const cells = Array.from(row.querySelectorAll('div, span'))
                    .filter(el => el.children.length === 0) // leaf nodes only
                    .map(el => el.innerText.trim())
                    .filter(t => t.length > 0);
                if (cells.length > 3) rows.push(cells);
            });

            return {
                containerClass: bestContainer.className,
                containerTag: bestContainer.tagName,
                childCount: bestCount,
                sampleRow0: rows[0] || [],
                sampleRow1: rows[1] || [],
                sampleRow2: rows[2] || [],
                rows: rows,
            };
        }
    """)


def extract_by_innertext(page, headers) -> list:
    """
    Nuclear option: grab the entire visible text of the table area,
    split into rows by known structure.
    Since we know the columns from the screenshot, parse accordingly.
    """
    return page.evaluate("""
        (headers) => {
            // Find all leaf text nodes that are visible and numeric or short strings
            // Strategy: find elements containing known team names
            const teamPattern = /^[A-Z][a-zA-Z .&'()-]{2,30}$/;
            const rows = [];

            // Look for repeated structures: find a parent that repeats
            // with one child matching a team name pattern
            const allEls = document.querySelectorAll('*');
            const rowEls = [];

            for (const el of allEls) {
                const text = el.innerText?.trim() || '';
                // A row element: direct text starts with a number (rank) or team name
                // and has several numeric siblings
                if (
                    el.children.length >= 5 &&
                    el.children.length <= 30 &&
                    /^\\d+$/.test(el.children[0]?.innerText?.trim())
                ) {
                    const cells = Array.from(el.children).map(c => c.innerText.trim());
                    rows.push(cells);
                }
            }
            return rows;
        }
    """, headers)


def scrape():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--no-sandbox"],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        print(f"Loading {TARGET_URL} …")
        page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(8_000)
        dismiss_popup(page)
        page.wait_for_timeout(3_000)

        # ── Dump raw HTML of the page so we can inspect it ──────────────────
        html = page.content()
        with open("evanmiya_page.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Saved full HTML → evanmiya_page.html ({len(html):,} bytes)")

        # ── Try rank-prefixed row extraction ────────────────────────────────
        print("\n  Trying rank-prefixed row extraction …")
        rows = extract_by_innertext(page, [])
        print(f"  Found {len(rows)} rows. Sample: {rows[:2]}")

        # ── Try div container extraction ─────────────────────────────────────
        print("\n  Trying div container extraction …")
        result = extract_div_table(page)
        if isinstance(result, dict):
            print(f"  Container class : {result.get('containerClass')}")
            print(f"  Container tag   : {result.get('containerTag')}")
            print(f"  Child count     : {result.get('childCount')}")
            print(f"  Sample row 0    : {result.get('sampleRow0')}")
            print(f"  Sample row 1    : {result.get('sampleRow1')}")
            div_rows = result.get('rows', [])
            print(f"  Total rows      : {len(div_rows)}")
        else:
            div_rows = result

        # ── Scroll and collect (for virtualised lists) ───────────────────────
        print("\n  Scrolling to collect all rows …")
        all_ranked_rows = _scroll_collect(page)
        print(f"  Scroll collected: {len(all_ranked_rows)} rows")

        browser.close()

    # Pick the best result
    best = max([rows, div_rows, all_ranked_rows], key=len)
    if not best:
        raise SystemExit("No rows found. Open evanmiya_page.html to inspect the structure.")

    print(f"\n✓ Using {len(best)} rows.")

    # If rows are lists (not dicts), we'll assign column names later
    if isinstance(best[0], list):
        df = pd.DataFrame(best)
    else:
        df = pd.DataFrame(best)

    df = clean(df)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved → {OUTPUT_CSV}")
    print(df.head(10).to_string())


def _scroll_collect(page) -> list:
    """Scroll down the page, collecting rank-prefixed rows each step."""
    seen = set()
    all_rows = []

    for step in range(80):
        rows = extract_by_innertext(page, [])
        for row in rows:
            key = str(row[:3])
            if key not in seen:
                seen.add(key)
                all_rows.append(row)
        if step % 10 == 0:
            print(f"    step {step}: {len(all_rows)} unique rows")
        page.evaluate("window.scrollBy(0, 400)")
        page.wait_for_timeout(200)

    return all_rows


def clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df.dropna(how="all", inplace=True)
    df = df[df.apply(lambda r: r.astype(str).str.strip().ne("").any(), axis=1)]
    df.dropna(axis=1, how="all", inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    scrape()
