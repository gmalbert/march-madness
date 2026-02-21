"""ROI Tracker – persist and analyse betting performance over time.

Usage:
    tracker = ROITracker()
    tracker.record_bet(date="2026-03-20", game="Duke vs UNC",
                       bet_type="moneyline", predicted_side="Duke",
                       actual_winner="Duke", american_odds=-150, stake=100)
    print(tracker.summary())
    tracker.save()
"""
import json
import os
import math
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Union, List, Dict, Any

DATA_DIR = Path("data_files")
DB_PATH = DATA_DIR / "roi_tracker.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal multiplier (profit per *unit* staked).
    Positive odds: e.g. +150 → profit 1.50 per unit
    Negative odds: e.g. -150 → profit 0.667 per unit
    """
    if odds >= 100:
        return odds / 100.0
    else:
        return 100.0 / abs(odds)


def bet_pnl(won: bool, stake: float, american_odds: float) -> float:
    """Return profit (+) or loss (-) for a single bet."""
    if won:
        return stake * american_to_decimal(american_odds)
    else:
        return -stake


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ROITracker:
    """Track betting picks, bankroll, and return-on-investment over time."""

    def __init__(self, db_path: Union[str, Path] = DB_PATH,
                 starting_bankroll: float = 1000.0):
        self.db_path = Path(db_path)
        self.starting_bankroll = starting_bankroll
        self.bets: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self.db_path.exists():
            with open(self.db_path, "r") as fh:
                data = json.load(fh)
            self.starting_bankroll = data.get("starting_bankroll", self.starting_bankroll)
            self.bets = data.get("bets", [])

    def save(self):
        """Persist bets to JSON file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as fh:
            json.dump({"starting_bankroll": self.starting_bankroll,
                       "bets": self.bets}, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_bet(
        self,
        *,
        game: str,
        bet_type: str,                    # "moneyline" | "spread" | "total"
        predicted_side: str,              # e.g. "Duke" or "Over"
        actual_result: str,               # e.g. "Duke" or "Over"
        american_odds: float,
        stake: float,
        date: Optional[Union[str, datetime]] = None,
        notes: str = "",
        model_confidence: Optional[float] = None,   # 0‑1
    ) -> Dict[str, Any]:
        """Record a single bet and return the entry dict."""

        won = (str(predicted_side).strip().lower() == str(actual_result).strip().lower())
        pnl = bet_pnl(won, stake, american_odds)
        entry = {
            "id": len(self.bets) + 1,
            "date": str(date) if date else datetime.today().strftime("%Y-%m-%d"),
            "game": game,
            "bet_type": bet_type,
            "predicted_side": predicted_side,
            "actual_result": actual_result,
            "american_odds": american_odds,
            "stake": stake,
            "won": won,
            "pnl": round(pnl, 2),
            "notes": notes,
            "model_confidence": model_confidence,
        }
        self.bets.append(entry)
        self.save()
        return entry

    def record_batch(self, bets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Record multiple bets at once."""
        return [self.record_bet(**b) for b in bets]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def _df(self):
        """Return bets as a pandas DataFrame (lazy import)."""
        import pandas as pd
        return pd.DataFrame(self.bets)

    def summary(self) -> Dict[str, Any]:
        """Return high-level performance summary."""
        if not self.bets:
            return {"total_bets": 0, "message": "No bets recorded yet."}

        total_bets = len(self.bets)
        wins = sum(1 for b in self.bets if b["won"])
        losses = total_bets - wins
        total_staked = sum(b["stake"] for b in self.bets)
        total_pnl = sum(b["pnl"] for b in self.bets)
        roi_pct = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        win_rate = wins / total_bets if total_bets > 0 else 0
        current_bankroll = self.starting_bankroll + total_pnl

        # By bet type
        bet_types: Dict[str, Dict] = {}
        for b in self.bets:
            bt = b.get("bet_type", "unknown")
            if bt not in bet_types:
                bet_types[bt] = {"bets": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}
            bet_types[bt]["bets"] += 1
            bet_types[bt]["wins"] += 1 if b["won"] else 0
            bet_types[bt]["pnl"] += b["pnl"]
            bet_types[bt]["staked"] += b["stake"]

        for bt_data in bet_types.values():
            bt_data["win_rate"] = bt_data["wins"] / bt_data["bets"] if bt_data["bets"] else 0
            bt_data["roi_pct"] = (bt_data["pnl"] / bt_data["staked"] * 100) if bt_data["staked"] else 0

        return {
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "total_staked": round(total_staked, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(roi_pct, 2),
            "starting_bankroll": self.starting_bankroll,
            "current_bankroll": round(current_bankroll, 2),
            "by_bet_type": bet_types,
        }

    def bankroll_history(self) -> List[Dict]:
        """Return bankroll after each bet in chronological order."""
        running = self.starting_bankroll
        history = []
        for b in self.bets:
            running += b["pnl"]
            history.append({
                "id": b["id"],
                "date": b["date"],
                "game": b["game"],
                "pnl": b["pnl"],
                "bankroll": round(running, 2),
            })
        return history

    def streak(self) -> Dict[str, int]:
        """Return current win/loss streak."""
        if not self.bets:
            return {"type": "none", "length": 0}
        streak_type = "win" if self.bets[-1]["won"] else "loss"
        length = 0
        for b in reversed(self.bets):
            if (b["won"] and streak_type == "win") or (not b["won"] and streak_type == "loss"):
                length += 1
            else:
                break
        return {"type": streak_type, "length": length}

    def by_month(self) -> Dict[str, Dict]:
        """Aggregate performance by calendar month."""
        months: Dict[str, Dict] = {}
        for b in self.bets:
            month = str(b.get("date", ""))[:7]  # "YYYY-MM"
            if month not in months:
                months[month] = {"bets": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}
            months[month]["bets"] += 1
            months[month]["wins"] += 1 if b["won"] else 0
            months[month]["pnl"] += b["pnl"]
            months[month]["staked"] += b["stake"]
        for m in months.values():
            m["roi_pct"] = (m["pnl"] / m["staked"] * 100) if m["staked"] else 0
            m["win_rate"] = m["wins"] / m["bets"] if m["bets"] else 0
        return months

    def best_model_confidence_threshold(self, min_confidence: float = 0.6) -> Dict:
        """Show performance only on bets where model confidence >= threshold."""
        high_conf = [b for b in self.bets
                     if b.get("model_confidence") is not None
                     and b["model_confidence"] >= min_confidence]
        if not high_conf:
            return {"message": f"No bets with confidence >= {min_confidence}"}
        wins = sum(1 for b in high_conf if b["won"])
        pnl = sum(b["pnl"] for b in high_conf)
        staked = sum(b["stake"] for b in high_conf)
        return {
            "min_confidence": min_confidence,
            "total_bets": len(high_conf),
            "wins": wins,
            "win_rate": wins / len(high_conf),
            "pnl": round(pnl, 2),
            "roi_pct": round(pnl / staked * 100, 2) if staked else 0,
        }

    def print_summary(self):
        """Pretty-print the tracker summary."""
        s = self.summary()
        print("\n" + "=" * 55)
        print("         BETTING ROI TRACKER SUMMARY")
        print("=" * 55)
        if s.get("total_bets", 0) == 0:
            print("  No bets recorded yet.")
            return
        print(f"  Bets      : {s['total_bets']:>6}  ({s['wins']}W – {s['losses']}L)")
        print(f"  Win Rate  : {s['win_rate']:.1%}")
        print(f"  Total PnL : ${s['total_pnl']:>+.2f}")
        print(f"  ROI       : {s['roi_pct']:>+.2f}%")
        print(f"  Bankroll  : ${s['starting_bankroll']:.2f} → ${s['current_bankroll']:.2f}")
        print()
        print("  By Bet Type:")
        for bt, d in s["by_bet_type"].items():
            print(f"    {bt:<12}: {d['bets']:>3} bets | WR {d['win_rate']:.1%} | ROI {d['roi_pct']:>+.1f}%")
        streak = self.streak()
        print(f"\n  Current streak: {streak['length']} {streak['type']}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# Historical backfill from existing predictions data
# ---------------------------------------------------------------------------

def backfill_from_historical(roi_tracker: "ROITracker",
                              csv_path: str = "data_files/historical_games_with_betting_predictions.csv",
                              stake: float = 100.0):
    """Load historical game results CSV and populate the ROI tracker.

    The CSV must have columns: game_date, home_team, away_team,
    pred_home_win_prob, home_win, home_moneyline, away_moneyline.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    required = ["home_team", "away_team", "pred_home_win_prob",
                "home_win", "home_moneyline", "away_moneyline"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"CSV missing columns: {missing}")
        return

    added = 0
    for _, row in df.iterrows():
        if pd.isna(row["pred_home_win_prob"]) or pd.isna(row["home_win"]):
            continue
        pred_home = row["pred_home_win_prob"] >= 0.5
        predicted_side = row["home_team"] if pred_home else row["away_team"]
        actual_winner = row["home_team"] if row["home_win"] == 1 else row["away_team"]
        odds = row["home_moneyline"] if pred_home else row["away_moneyline"]
        if pd.isna(odds):
            continue
        game_date = row.get("game_date", "")
        roi_tracker.record_bet(
            game=f"{row['away_team']} @ {row['home_team']}",
            bet_type="moneyline",
            predicted_side=predicted_side,
            actual_result=actual_winner,
            american_odds=float(odds),
            stake=stake,
            date=game_date,
            model_confidence=float(row["pred_home_win_prob"]) if pred_home else 1 - float(row["pred_home_win_prob"]),
        )
        added += 1
    print(f"Backfilled {added} historical bets into ROI tracker.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tracker = ROITracker()

    # Demo: record a handful of sample bets
    sample_bets = [
        dict(game="Duke vs North Carolina", bet_type="moneyline",
             predicted_side="Duke", actual_result="Duke",
             american_odds=-180, stake=100, date="2026-03-01",
             model_confidence=0.72),
        dict(game="Kansas vs Kentucky", bet_type="spread",
             predicted_side="Kansas -3.5", actual_result="Kansas -3.5",
             american_odds=-110, stake=100, date="2026-03-02",
             model_confidence=0.61),
        dict(game="Gonzaga vs Arizona", bet_type="moneyline",
             predicted_side="Gonzaga", actual_result="Arizona",
             american_odds=-140, stake=100, date="2026-03-05",
             model_confidence=0.58),
        dict(game="Auburn vs Tennessee", bet_type="total",
             predicted_side="Over 147.5", actual_result="Over 147.5",
             american_odds=-110, stake=50, date="2026-03-07",
             model_confidence=0.65),
    ]
    tracker.record_batch(sample_bets)
    tracker.print_summary()
    print("\nBankroll history:")
    for entry in tracker.bankroll_history():
        print(f"  {entry['date']}  {entry['game']:<45}  PnL: ${entry['pnl']:>+7.2f}  Bankroll: ${entry['bankroll']:.2f}")
