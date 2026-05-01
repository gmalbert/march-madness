"""
scripts/export_best_bets.py — March Madness / NCAAB (march-madness)
Reads data_files/upcoming_game_predictions.json (precomputed nightly) and writes
data_files/best_bets_today.json in the unified Sports Picks Grid schema.
"""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SPORT = "NCAAB"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH = Path("data_files/best_bets_today.json")
PREDS_PATH = Path("data_files/upcoming_game_predictions.json")
LOOKAHEAD_DAYS = 7  # Tournament spans full week


def _write(bets: list, notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _tier_from_ci(ci_width: float) -> str:
    if ci_width <= 4:
        return "Elite"
    elif ci_width <= 8:
        return "Strong"
    elif ci_width <= 14:
        return "Good"
    return "Standard"


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> None:
    today = date.today()

    # NCAAB active season: November–April
    month = today.month
    if not (11 <= month or month <= 4):
        _write([], "NCAAB off-season")
        return

    if not PREDS_PATH.exists():
        _write([], "upcoming_game_predictions.json not found — run precompute pipeline first")
        return

    try:
        with open(PREDS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        _write([], f"Failed to read predictions JSON: {e}")
        return

    # Support both list and dict with a "games" / "predictions" key
    if isinstance(raw, list):
        games = raw
    elif isinstance(raw, dict):
        games = raw.get("games", raw.get("predictions", raw.get("bets", [])))
    else:
        _write([], "Unexpected JSON format")
        return

    if not games:
        _write([], f"No NCAAB game predictions available")
        return

    end_date = today + timedelta(days=LOOKAHEAD_DAYS)
    bets = []

    for g in games:
        game_date_str = str(g.get("date", g.get("game_date", "")))
        try:
            gd = date.fromisoformat(game_date_str[:10])
        except (ValueError, TypeError):
            continue

        if not (today <= gd <= end_date):
            continue

        home = str(g.get("home_team", ""))
        away = str(g.get("away_team", ""))
        spread = _safe_float(g.get("spread_prediction", g.get("predicted_spread")))
        ci_high = _safe_float(g.get("ci_high", g.get("confidence_high")))
        ci_low  = _safe_float(g.get("ci_low",  g.get("confidence_low")))
        ml_home = g.get("home_moneyline")
        ml_away = g.get("away_moneyline")

        # CI-based tier
        if ci_high is not None and ci_low is not None:
            ci_width = abs(ci_high - ci_low)
        else:
            ci_width = 14  # unknown — default to "Good"

        tier = _tier_from_ci(ci_width)
        if tier == "Standard":
            continue  # Too uncertain

        # Confidence as inverse of CI width (normalized 0-1)
        conf = max(0.50, min(0.99, 1.0 - ci_width / 30.0)) if ci_high is not None else 0.60

        # Derive pick from spread
        if spread is not None:
            if spread < 0:
                pick = f"{home} {spread:.1f}"
                odds = ml_home
            else:
                pick = f"{away} +{abs(spread):.1f}" if spread > 0 else f"{home} PK"
                odds = ml_away
            bet_type = "Spread"
        else:
            # Moneyline fallback
            bet_type = "Moneyline"
            pick = home
            odds = ml_home

        # Notes field
        notes_parts = []
        if g.get("model_source"):
            notes_parts.append(str(g["model_source"]))
        if ci_high is not None and ci_low is not None:
            notes_parts.append(f"CI: [{ci_low:.1f}, {ci_high:.1f}]")
        notes_str = " | ".join(notes_parts) if notes_parts else None

        bet: dict = {
            "game_date": game_date_str[:10],
            "game_time": str(g.get("game_time", "")) or None,
            "game": f"{away} @ {home}",
            "home_team": home,
            "away_team": away,
            "bet_type": bet_type,
            "pick": pick,
            "confidence": round(conf, 4),
            "edge": None,
            "tier": tier,
            "odds": int(odds) if odds and str(odds) not in ("nan", "None", "") else None,
            "line": spread,
            "league": "NCAAB",
            "notes": notes_str,
        }
        bets.append(bet)

    _write(bets, "" if bets else f"No qualifying NCAAB picks in next {LOOKAHEAD_DAYS} days")


if __name__ == "__main__":
    main()
