"""Export Picks – write betting recommendations to CSV / JSON / PDF.

Usage:
    picks = [
        {"game": "Duke vs UNC", "bet_type": "moneyline", "pick": "Duke",
         "confidence": 0.72, "american_odds": -150, "kelly_stake_pct": 0.04},
        ...
    ]
    export_picks_to_csv(picks, "my_picks.csv")
    export_picks_to_json(picks, "my_picks.json")
    export_picks_to_pdf(picks, "my_picks.pdf")   # requires reportlab
    export_picks_to_html(picks, "my_picks.html")
"""
import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

EXPORT_DIR = Path("data_files") / "exported_picks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _timestamp_filename(base: str, ext: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR / f"{base}_{ts}.{ext}"


def _normalise_picks(picks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill in missing keys with sensible defaults so export is consistent."""
    defaults = {
        "date": datetime.today().strftime("%Y-%m-%d"),
        "game": "Unknown",
        "bet_type": "moneyline",
        "pick": "N/A",
        "confidence": None,
        "american_odds": None,
        "kelly_stake_pct": None,
        "notes": "",
    }
    return [{**defaults, **p} for p in picks]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_picks_to_csv(picks: List[Dict[str, Any]],
                        filename: Optional[Union[str, Path]] = None) -> str:
    """Export betting picks to a CSV file.

    Returns the path to the written file.
    """
    picks = _normalise_picks(picks)
    if not picks:
        raise ValueError("picks list is empty")

    out_path = Path(filename) if filename else _timestamp_filename("picks", "csv")
    _ensure_dir(out_path)

    fieldnames = list(picks[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(picks)

    print(f"Picks exported → {out_path}  ({len(picks)} rows)")
    return str(out_path)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def export_picks_to_json(picks: List[Dict[str, Any]],
                         filename: Optional[Union[str, Path]] = None) -> str:
    """Export betting picks to a JSON file."""
    picks = _normalise_picks(picks)
    if not picks:
        raise ValueError("picks list is empty")

    out_path = Path(filename) if filename else _timestamp_filename("picks", "json")
    _ensure_dir(out_path)

    payload = {
        "exported_at": datetime.now().isoformat(),
        "total_picks": len(picks),
        "picks": picks,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"Picks exported → {out_path}  ({len(picks)} picks)")
    return str(out_path)


# ---------------------------------------------------------------------------
# HTML (lightweight, no extra deps)
# ---------------------------------------------------------------------------

def export_picks_to_html(picks: List[Dict[str, Any]],
                         filename: Optional[Union[str, Path]] = None,
                         title: str = "March Madness Betting Picks") -> str:
    """Export picks to a self-contained HTML file with a styled table."""
    picks = _normalise_picks(picks)
    if not picks:
        raise ValueError("picks list is empty")

    out_path = Path(filename) if filename else _timestamp_filename("picks", "html")
    _ensure_dir(out_path)

    columns = list(picks[0].keys())

    def _cell(val: Any) -> str:
        if val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    rows_html = ""
    for p in picks:
        conf = p.get("confidence")
        colour = ""
        if conf is not None:
            if conf >= 0.65:
                colour = "background:#d4edda"
            elif conf >= 0.55:
                colour = "background:#fff3cd"
            else:
                colour = "background:#f8d7da"
        cells = "".join(f"<td>{_cell(p.get(c))}</td>" for c in columns)
        rows_html += f"<tr style='{colour}'>{cells}</tr>\n"

    header_html = "".join(f"<th>{c}</th>" for c in columns)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ color: #2c3e50; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    th {{ background: #2c3e50; color: white; }}
    tr:hover {{ opacity: 0.85; }}
    .footer {{ color: #888; margin-top: 10px; font-size: 0.8em; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Generated: {ts} &nbsp;|&nbsp; Total picks: {len(picks)}</p>
  <table>
    <thead><tr>{header_html}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p class="footer">Green = confidence ≥ 65% | Yellow = 55‑64% | Red = &lt; 55%</p>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Picks exported → {out_path}  ({len(picks)} picks)")
    return str(out_path)


# ---------------------------------------------------------------------------
# PDF (uses reportlab if available; falls back to HTML)
# ---------------------------------------------------------------------------

def export_picks_to_pdf(picks: List[Dict[str, Any]],
                        filename: Optional[Union[str, Path]] = None,
                        title: str = "March Madness Betting Picks") -> str:
    """Export picks to PDF.  Requires *reportlab*; falls back to HTML if absent."""
    try:
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        return _export_pdf_reportlab(picks, filename, title)
    except ImportError:
        print("reportlab not installed – falling back to HTML export.")
        html_path = str(out_path).replace(".pdf", ".html") if filename else None
        return export_picks_to_html(picks, html_path, title)


def _export_pdf_reportlab(picks, filename, title):
    """Internal PDF export using reportlab."""
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    picks = _normalise_picks(picks)
    out_path = Path(filename) if filename else _timestamp_filename("picks", "pdf")
    _ensure_dir(out_path)

    doc = SimpleDocTemplate(str(out_path), pagesize=landscape(letter),
                             leftMargin=30, rightMargin=30,
                             topMargin=40, bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(title, styles["Title"]))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    elements.append(Paragraph(f"Generated: {ts}  ·  {len(picks)} picks", styles["Normal"]))
    elements.append(Spacer(1, 12))

    columns = list(picks[0].keys())
    header_row = [c.replace("_", " ").title() for c in columns]
    table_data = [header_row] + [[str(p.get(c, "")) for c in columns] for p in picks]

    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(tbl)

    doc.build(elements)
    print(f"Picks exported → {out_path}  ({len(picks)} picks)")
    return str(out_path)


# ---------------------------------------------------------------------------
# Convenience: generate picks from predictions JSON
# ---------------------------------------------------------------------------

def picks_from_predictions_json(
    path: str = "data_files/upcoming_game_predictions.json",
    min_edge: float = 0.03,
    bet_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract betting picks from the precomputed predictions JSON.

    Returns a list of pick dicts where model edge >= min_edge.
    """
    with open(path, "r", encoding="utf-8") as fh:
        games = json.load(fh)

    if bet_types is None:
        bet_types = ["moneyline", "spread", "total"]

    picks = []
    for g in games:
        game_str = f"{g.get('away_team', '?')} @ {g.get('home_team', '?')}"
        game_date = g.get("game_date", g.get("date", ""))

        # Moneyline pick
        if "moneyline" in bet_types:
            ml_prob = g.get("home_win_probability") or g.get("pred_home_win_prob")
            home_ml = g.get("home_moneyline")
            away_ml = g.get("away_moneyline")
            if ml_prob is not None and home_ml is not None and away_ml is not None:
                if ml_prob >= 0.5:
                    implied = 100 / (abs(home_ml) + 100) if home_ml < 0 else home_ml / (home_ml + 100)
                    edge = ml_prob - implied
                    side = g.get("home_team", "Home")
                    odds = home_ml
                else:
                    implied = 100 / (abs(away_ml) + 100) if away_ml < 0 else away_ml / (away_ml + 100)
                    edge = (1 - ml_prob) - implied
                    side = g.get("away_team", "Away")
                    odds = away_ml
                if edge >= min_edge:
                    picks.append({
                        "date": game_date,
                        "game": game_str,
                        "bet_type": "moneyline",
                        "pick": side,
                        "confidence": round(float(ml_prob if ml_prob >= 0.5 else 1 - ml_prob), 4),
                        "american_odds": odds,
                        "edge_pct": round(edge * 100, 2),
                        "notes": "",
                    })

        # Spread pick
        if "spread" in bet_types:
            pred_spread = g.get("predicted_spread") or g.get("pred_spread")
            bet_spread = g.get("betting_spread")
            if pred_spread is not None and bet_spread is not None:
                # If predicted margin covers the spread, bet the home team -spread
                if (pred_spread - float(bet_spread)) > 2:  # buffer
                    picks.append({
                        "date": game_date,
                        "game": game_str,
                        "bet_type": "spread",
                        "pick": f"{g.get('home_team')} {bet_spread:+.1f}",
                        "confidence": None,
                        "american_odds": -110,
                        "edge_pct": None,
                        "notes": f"pred margin {pred_spread:+.1f} vs spread {bet_spread:+.1f}",
                    })

        # Total pick
        if "total" in bet_types:
            pred_total = g.get("predicted_total") or g.get("pred_total")
            bet_total = g.get("betting_over_under") or g.get("over_under")
            if pred_total is not None and bet_total is not None:
                side = "Over" if pred_total > float(bet_total) else "Under"
                if abs(pred_total - float(bet_total)) > 2:  # buffer
                    picks.append({
                        "date": game_date,
                        "game": game_str,
                        "bet_type": "total",
                        "pick": f"{side} {bet_total}",
                        "confidence": None,
                        "american_odds": -110,
                        "edge_pct": None,
                        "notes": f"pred total {pred_total:.1f}",
                    })

    return picks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    predictions_path = "data_files/upcoming_game_predictions.json"
    if not Path(predictions_path).exists():
        print(f"No predictions file found at {predictions_path}. Using sample data.")
        picks = [
            {"game": "Duke vs UNC", "bet_type": "moneyline", "pick": "Duke",
             "confidence": 0.72, "american_odds": -165, "edge_pct": 5.2},
            {"game": "Kansas vs Kentucky", "bet_type": "spread", "pick": "Kansas -3.5",
             "confidence": 0.61, "american_odds": -110, "edge_pct": None},
            {"game": "Gonzaga vs Arizona", "bet_type": "total", "pick": "Over 147.5",
             "confidence": None, "american_odds": -110, "edge_pct": 3.1},
        ]
    else:
        picks = picks_from_predictions_json(predictions_path)
        print(f"Found {len(picks)} picks with edge >= 3%")

    if picks:
        csv_path = export_picks_to_csv(picks)
        json_path = export_picks_to_json(picks)
        html_path = export_picks_to_html(picks)
        print(f"\nExported {len(picks)} picks:")
        print(f"  CSV  → {csv_path}")
        print(f"  JSON → {json_path}")
        print(f"  HTML → {html_path}")
    else:
        print("No picks to export.")
