"""
evanmiya.com DOM Diagnostic
============================
Run this FIRST to figure out how the table is structured,
then we can fix the main scraper accordingly.

Output: evanmiya_debug.html  (full page HTML snapshot)
        evanmiya_debug.txt   (summary of found elements)
"""

from playwright.sync_api import sync_playwright

TARGET_URL = "https://evanmiya.com/?team_ratings"


def debug():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,  # visible so you can see what loads
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

        print("Loading page...")
        page.goto(TARGET_URL, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(5000)  # extra wait for JS hydration

        # ── 1. Save full HTML snapshot ────────────────────────────────────────
        html = page.content()
        with open("evanmiya_debug.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✓ Full HTML saved to evanmiya_debug.html ({len(html):,} bytes)")

        # ── 2. Probe the DOM for known table patterns ─────────────────────────
        report = page.evaluate("""
            () => {
                const info = {};

                // Standard elements
                info.tables        = document.querySelectorAll('table').length;
                info.tbodies       = document.querySelectorAll('tbody').length;
                info.tr_count      = document.querySelectorAll('tr').length;
                info.th_count      = document.querySelectorAll('th').length;
                info.td_count      = document.querySelectorAll('td').length;

                // Role-based (ARIA)
                info.role_table    = document.querySelectorAll('[role="table"]').length;
                info.role_row      = document.querySelectorAll('[role="row"]').length;
                info.role_cell     = document.querySelectorAll('[role="cell"]').length;
                info.role_colhdr   = document.querySelectorAll('[role="columnheader"]').length;
                info.role_grid     = document.querySelectorAll('[role="grid"]').length;
                info.role_gridcell = document.querySelectorAll('[role="gridcell"]').length;

                // React / AG Grid / Tanstack
                info.rt_table      = document.querySelectorAll('.rt-table').length;
                info.rt_tr         = document.querySelectorAll('.rt-tr').length;
                info.rt_td         = document.querySelectorAll('.rt-td').length;
                info.ag_root       = document.querySelectorAll('.ag-root').length;
                info.ag_row        = document.querySelectorAll('.ag-row').length;

                // Generic div rows (some sites use pure divs)
                info.divs_total    = document.querySelectorAll('div').length;

                // First <table> outer HTML snippet (first 500 chars)
                const t = document.querySelector('table');
                info.first_table_snippet = t ? t.outerHTML.slice(0, 500) : 'NO TABLE FOUND';

                // All unique class names on <tr> or role=row elements
                const rows = document.querySelectorAll('tr, [role="row"]');
                const classSet = new Set();
                rows.forEach(r => r.classList.forEach(c => classSet.add(c)));
                info.row_classes = [...classSet].slice(0, 30);

                // page title
                info.title = document.title;

                // Any element containing text like "Duke" or "Michigan" (likely a data cell)
                const allText = document.body.innerText;
                info.contains_duke    = allText.includes('Duke');
                info.contains_michigan = allText.includes('Michigan');
                info.body_text_snippet = allText.slice(0, 800);

                return info;
            }
        """)

        # ── 3. Print report ───────────────────────────────────────────────────
        lines = [
            "=" * 60,
            f"Page title: {report['title']}",
            "",
            "── Standard HTML ──",
            f"  <table> elements : {report['tables']}",
            f"  <tbody> elements : {report['tbodies']}",
            f"  <tr> elements    : {report['tr_count']}",
            f"  <th> elements    : {report['th_count']}",
            f"  <td> elements    : {report['td_count']}",
            "",
            "── ARIA roles ──",
            f"  role=table       : {report['role_table']}",
            f"  role=row         : {report['role_row']}",
            f"  role=cell        : {report['role_cell']}",
            f"  role=columnheader: {report['role_colhdr']}",
            f"  role=grid        : {report['role_grid']}",
            f"  role=gridcell    : {report['role_gridcell']}",
            "",
            "── React Table / AG Grid ──",
            f"  .rt-table        : {report['rt_table']}",
            f"  .rt-tr           : {report['rt_tr']}",
            f"  .rt-td           : {report['rt_td']}",
            f"  .ag-root         : {report['ag_root']}",
            f"  .ag-row          : {report['ag_row']}",
            "",
            "── Content check ──",
            f"  Contains 'Duke'    : {report['contains_duke']}",
            f"  Contains 'Michigan': {report['contains_michigan']}",
            "",
            "── Row CSS classes found ──",
            "  " + str(report['row_classes']),
            "",
            "── First <table> snippet ──",
            report['first_table_snippet'],
            "",
            "── Body text snippet ──",
            report['body_text_snippet'],
            "=" * 60,
        ]
        output = "\n".join(lines)
        print(output)

        with open("evanmiya_debug.txt", "w", encoding="utf-8") as f:
            f.write(output)
        print("\n✓ Summary saved to evanmiya_debug.txt")

        input("\nPress ENTER to close the browser...")
        browser.close()


if __name__ == "__main__":
    debug()