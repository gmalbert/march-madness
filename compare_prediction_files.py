import json
import os
from datetime import datetime

files = {
    'precomputed': 'data_files/precomputed_predictions/predictions_2026-02-03.json',
    'upcoming': 'data_files/upcoming_game_predictions.json'
}

for name, path in files.items():
    if os.path.exists(path):
        data = json.load(open(path))
        games = data if isinstance(data, list) else data.get('games', [])
        
        with_spread = sum(1 for g in games if g.get('game_info', {}).get('home_spread') is not None)
        with_total = sum(1 for g in games if g.get('game_info', {}).get('total_line') is not None)
        
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        
        print(f"\n{'='*60}")
        print(f"{name.upper()}: {path}")
        print(f"{'='*60}")
        print(f"Total games:        {len(games)}")
        print(f"Games with spread:  {with_spread} ({with_spread/len(games)*100:.1f}%)")
        print(f"Games with total:   {with_total} ({with_total/len(games)*100:.1f}%)")
        print(f"Last modified:      {mtime}")
        
        if with_spread > 0:
            print(f"\nSample game with betting data:")
            sample = next((g for g in games if g.get('game_info', {}).get('home_spread') is not None), None)
            if sample:
                gi = sample['game_info']
                print(f"  {gi.get('away_team', 'Unknown')} @ {gi.get('home_team', 'Unknown')}")
                print(f"  Spread: {gi.get('home_spread')}, Total: {gi.get('total_line')}")
    else:
        print(f"\n{name.upper()}: NOT FOUND - {path}")
