#!/usr/bin/env python3
"""Check upcoming ESPN games for missing efficiency metrics or model predictions.

Usage: python scripts/check_upcoming_games.py
"""

import sys
from pathlib import Path
# ensure project root is on sys.path so local packages can be imported
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import importlib
import pandas as pd
import predictions
importlib.reload(predictions)


def main():
    games = predictions.get_upcoming_games()
    missing = []
    for g in games:
        home = g.get('home_team') or g.get('home_name') or g.get('home')
        away = g.get('away_team') or g.get('away_name') or g.get('away')
        home_vals = [g.get(k) for k in ['home_kenpom','home_bart','home_eff','home_stats','home_efficiency','home_kenpom_net','home_bart_adj']]
        away_vals = [g.get(k) for k in ['away_kenpom','away_bart','away_eff','away_stats','away_efficiency','away_kenpom_net','away_bart_adj']]
        home_has = any(pd.notna(x) for x in home_vals)
        away_has = any(pd.notna(x) for x in away_vals)
        pred = None
        for k in ['predicted_spread','prediction','spread_prediction','spread_pred','model_prediction']:
            if k in g:
                pred = g[k]
                break
        if (pred is None) or (isinstance(pred, str) and pred.strip().upper()=='N/A') or (not home_has) or (not away_has):
            missing.append({'home': home, 'away': away, 'pred': pred, 'home_has': home_has, 'away_has': away_has})

    print(f"Total upcoming games: {len(games)}")
    print(f"Missing or incomplete: {len(missing)}")
    for m in missing:
        print(f"- {m['home']} vs {m['away']} | pred={m['pred']} | home_has={m['home_has']} | away_has={m['away_has']}")


if __name__ == '__main__':
    main()
