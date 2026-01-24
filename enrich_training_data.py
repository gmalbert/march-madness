"""
Enrich historical training data with KenPom and BartTorvik features.

This script takes the existing training_data_weighted.csv and adds KenPom/BartTorvik
efficiency ratings where available, creating an extended feature set for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_tools.efficiency_loader import EfficiencyDataLoader

DATA_DIR = Path("data_files")

# Import normalize_team_name from predictions.py
import sys
sys.path.append('.')
from predictions import normalize_team_name


def enrich_training_data():
    """Add KenPom and BartTorvik features to historical training data."""
    
    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "training_data_weighted.csv")
    print(f"Loaded {len(df)} games from {sorted(df['season'].unique())}")
    
    print("\nLoading KenPom and BartTorvik data...")
    loader = EfficiencyDataLoader()
    
    try:
        kenpom_df = loader.load_kenpom()
        print(f"Loaded KenPom data: {len(kenpom_df)} teams")
    except Exception as e:
        print(f"Warning: Could not load KenPom data: {e}")
        kenpom_df = None
    
    try:
        bart_df = loader.load_barttorvik()
        print(f"Loaded BartTorvik data: {len(bart_df)} teams")
    except Exception as e:
        print(f"Warning: Could not load BartTorvik data: {e}")
        bart_df = None
    
    if kenpom_df is None and bart_df is None:
        print("Error: No advanced metrics available. Exiting.")
        return
    
    # Initialize new columns
    kenpom_features = ['kenpom_netrtg_diff', 'kenpom_ortg_diff', 'kenpom_drtg_diff', 
                       'kenpom_adjt_diff', 'kenpom_luck_diff', 'kenpom_sos_diff']
    bart_features = ['bart_oe_diff', 'bart_de_diff']
    
    for col in kenpom_features + bart_features:
        df[col] = np.nan
    
    print("\nEnriching games with advanced metrics...")
    matched_games = 0
    kenpom_matches = 0
    bart_matches = 0
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} games...")
        
        # Normalize team names
        home_team = normalize_team_name(row['home_team'])
        away_team = normalize_team_name(row['away_team'])
        
        # Get KenPom data
        if kenpom_df is not None:
            home_kp = kenpom_df[kenpom_df['canonical_team'] == home_team]
            away_kp = kenpom_df[kenpom_df['canonical_team'] == away_team]
            
            if not home_kp.empty and not away_kp.empty:
                home_kp = home_kp.iloc[0]
                away_kp = away_kp.iloc[0]
                
                df.at[idx, 'kenpom_netrtg_diff'] = home_kp['NetRtg'] - away_kp['NetRtg']
                df.at[idx, 'kenpom_ortg_diff'] = home_kp['ORtg'] - away_kp['ORtg']
                df.at[idx, 'kenpom_drtg_diff'] = home_kp['DRtg'] - away_kp['DRtg']
                df.at[idx, 'kenpom_adjt_diff'] = home_kp['AdjT'] - away_kp['AdjT']
                df.at[idx, 'kenpom_luck_diff'] = home_kp['Luck'] - away_kp['Luck']
                df.at[idx, 'kenpom_sos_diff'] = home_kp['SOS_NetRtg'] - away_kp['SOS_NetRtg']
                
                kenpom_matches += 1
        
        # Get BartTorvik data
        if bart_df is not None:
            home_bt = bart_df[bart_df['canonical_team'] == home_team]
            away_bt = bart_df[bart_df['canonical_team'] == away_team]
            
            if not home_bt.empty and not away_bt.empty:
                home_bt = home_bt.iloc[0]
                away_bt = away_bt.iloc[0]
                
                df.at[idx, 'bart_oe_diff'] = home_bt['Adj OE'] - away_bt['Adj OE']
                df.at[idx, 'bart_de_diff'] = home_bt['Adj DE'] - away_bt['Adj DE']
                
                bart_matches += 1
        
        # Track if any advanced metrics were found
        if kenpom_matches > 0 or bart_matches > 0:
            matched_games += 1
    
    print(f"\nEnrichment complete:")
    print(f"  Total games: {len(df)}")
    print(f"  Games with KenPom data: {kenpom_matches} ({kenpom_matches/len(df)*100:.1f}%)")
    print(f"  Games with BartTorvik data: {bart_matches} ({bart_matches/len(df)*100:.1f}%)")
    print(f"  Games with any advanced metrics: {matched_games} ({matched_games/len(df)*100:.1f}%)")
    
    # Check feature availability
    print("\nFeature availability:")
    for col in kenpom_features + bart_features:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null} ({non_null/len(df)*100:.1f}%)")
    
    # Save enriched data
    output_path = DATA_DIR / "training_data_enriched.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved enriched training data to {output_path}")
    
    # Also create a version with only complete cases (all features available)
    complete_cases = df.dropna(subset=kenpom_features + bart_features)
    if len(complete_cases) > 0:
        complete_path = DATA_DIR / "training_data_complete_features.csv"
        complete_cases.to_csv(complete_path, index=False)
        print(f"✅ Saved {len(complete_cases)} complete cases to {complete_path}")
    else:
        print("⚠️ No games have complete advanced metrics (expected - using current season data)")
        print("   Will use partial enrichment for training")


if __name__ == '__main__':
    enrich_training_data()
