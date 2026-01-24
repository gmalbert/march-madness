import importlib
import pandas as pd
import predictions
importlib.reload(predictions)

games = predictions.get_upcoming_games()
missing = []
for g in games:
    home = g.get('home_team') or g.get('home_name') or g.get('home') or g.get('home_school') or g.get('home_alias')
    away = g.get('away_team') or g.get('away_name') or g.get('away') or g.get('away_school') or g.get('away_alias')
    pred = None
    for k in ['predicted_spread','spread_pred','prediction','pred_spread','model_prediction','spread_prediction']:
        if k in g:
            pred = g[k]
            break
    home_vals = [g.get(k) for k in ['home_kenpom','home_kenpom_net','home_kenpom_adj','home_bart','home_bart_adj','home_bart_value','home_kenpom_value']]
    away_vals = [g.get(k) for k in ['away_kenpom','away_kenpom_net','away_kenpom_adj','away_bart','away_bart_adj','away_bart_value','away_kenpom_value']]
    home_has = any(pd.notna(x) for x in home_vals)
    away_has = any(pd.notna(x) for x in away_vals)
    if (pred is None) or (isinstance(pred, str) and pred.strip().upper()=='N/A') or (not home_has) or (not away_has):
        missing.append((home, away, pred, home_has, away_has))

print(len(games), 'games scanned')
print(len(missing), 'missing or incomplete:')
for h, a, p, hp, ap in missing:
    print(f"{h} vs {a} | pred={p} | home_has={hp} | away_has={ap}")
