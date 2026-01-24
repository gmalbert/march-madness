"""
KenPom and BartTorvik data loader with canonical team name mapping.

Loads efficiency ratings from KenPom and BartTorvik CSVs, applies canonical
team name mappings, cleans numeric columns, and outputs merged datasets
ready for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class EfficiencyDataLoader:
    """Load and merge KenPom and BartTorvik efficiency data."""
    
    def __init__(self, data_dir='data_files'):
        self.data_dir = Path(data_dir)
        self.kenpom_path = self.data_dir / 'kenpom_ratings.csv'
        self.bart_path = self.data_dir / 'barttorvik_ratings.csv'
        self.kenpom_map_path = self.data_dir / 'kenpom_to_espn_matches.csv'
        self.bart_map_path = self.data_dir / 'bart_to_espn_matches.csv'
    
    def load_kenpom(self):
        """Load and clean KenPom ratings with canonical team names."""
        df = pd.read_csv(self.kenpom_path)
        mappings = pd.read_csv(self.kenpom_map_path)
        
        # Apply canonical team name mapping
        team_map = dict(zip(mappings['kenpom'], mappings['espn_match']))
        df['canonical_team'] = df['Team'].map(team_map)
        
        # Drop unmapped teams
        unmapped = df[df['canonical_team'].isna()]
        if len(unmapped) > 0:
            print(f"Warning: Dropping {len(unmapped)} unmapped KenPom teams: {unmapped['Team'].tolist()}")
        df = df[df['canonical_team'].notna()].copy()
        
        # Clean numeric columns (handle + signs)
        def clean_numeric(series):
            if series.dtype == 'object':
                return series.str.replace('+', '', regex=False).astype(float)
            return pd.to_numeric(series, errors='coerce')
        
        df['NetRtg'] = clean_numeric(df['NetRtg'])
        df['ORtg'] = clean_numeric(df['ORtg'])
        df['DRtg'] = clean_numeric(df['DRtg'])
        df['AdjT'] = clean_numeric(df['AdjT'])
        df['Luck'] = clean_numeric(df['Luck'])
        df['SOS_NetRtg'] = clean_numeric(df['SOS_NetRtg'])
        df['SOS_ORtg'] = clean_numeric(df['SOS_ORtg'])
        df['SOS_DRtg'] = clean_numeric(df['SOS_DRtg'])
        df['NCSOS_NetRtg'] = clean_numeric(df['NCSOS_NetRtg'])
        
        # Clean rank columns
        for col in df.columns:
            if col.endswith('_Rank'):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Parse W-L into wins and losses
        wl_split = df['W-L'].str.split('-', expand=True)
        df['wins'] = wl_split[0].astype(int)
        df['losses'] = wl_split[1].astype(int)
        df['win_pct'] = df['wins'] / (df['wins'] + df['losses'])
        
        # Add source identifier
        df['source'] = 'kenpom'
        
        return df
    
    def load_barttorvik(self):
        """Load and clean BartTorvik ratings with canonical team names."""
        df = pd.read_csv(self.bart_path)
        mappings = pd.read_csv(self.bart_map_path)
        
        # Apply canonical team name mapping
        team_map = dict(zip(mappings['bart'], mappings['espn_match']))
        df['canonical_team'] = df['Team'].map(team_map)
        
        # Drop unmapped teams
        unmapped = df[df['canonical_team'].isna()]
        if len(unmapped) > 0:
            print(f"Warning: Dropping {len(unmapped)} unmapped BartTorvik teams: {unmapped['Team'].tolist()}")
        df = df[df['canonical_team'].notna()].copy()
        
        # Clean numeric columns (BartTorvik data is already clean)
        # Parse Record if needed
        if 'Record' in df.columns:
            record_split = df['Record'].str.split('–', expand=True)
            if record_split.shape[1] == 2:
                df['bart_wins'] = pd.to_numeric(record_split[0], errors='coerce')
                df['bart_losses'] = pd.to_numeric(record_split[1], errors='coerce')
        
        # Add source identifier
        df['source'] = 'barttorvik'
        
        return df
    
    def get_kenpom_for_team(self, team_name):
        """Get KenPom data for a specific canonical team name."""
        kp = self.load_kenpom()
        team_data = kp[kp['canonical_team'] == team_name]
        if team_data.empty:
            return None
        return team_data.iloc[0].to_dict()
    
    def get_barttorvik_for_team(self, team_name):
        """Get BartTorvik data for a specific canonical team name."""
        bt = self.load_barttorvik()
        team_data = bt[bt['canonical_team'] == team_name]
        if team_data.empty:
            return None
        return team_data.iloc[0].to_dict()
    
    def get_merged_efficiency(self, team_name):
        """Get merged efficiency data from both sources for a team."""
        kp_data = self.get_kenpom_for_team(team_name)
        bt_data = self.get_barttorvik_for_team(team_name)
        
        return {
            'team': team_name,
            'kenpom': kp_data,
            'barttorvik': bt_data
        }
    
    def save_canonical_datasets(self):
        """Save cleaned, canonicalized datasets for easy reuse."""
        kp = self.load_kenpom()
        bt = self.load_barttorvik()
        
        kp.to_csv(self.data_dir / 'kenpom_canonical.csv', index=False)
        bt.to_csv(self.data_dir / 'barttorvik_canonical.csv', index=False)
        
        print(f"Saved {len(kp)} KenPom teams to kenpom_canonical.csv")
        print(f"Saved {len(bt)} BartTorvik teams to barttorvik_canonical.csv")
        
        return kp, bt


def main():
    """Example usage of the efficiency data loader."""
    loader = EfficiencyDataLoader()
    
    # Load and save canonical datasets
    print("Loading and cleaning efficiency data...")
    kp, bt = loader.save_canonical_datasets()
    
    # Example: Get data for a specific team
    print("\nExample: Michigan data")
    michigan_data = loader.get_merged_efficiency('Michigan')
    if michigan_data['kenpom']:
        print(f"  KenPom NetRtg: {michigan_data['kenpom']['NetRtg']}")
        print(f"  KenPom Rank: {michigan_data['kenpom']['Rk']}")
    if michigan_data['barttorvik']:
        print(f"  BartTorvik Adj OE: {michigan_data['barttorvik']['Adj OE']}")
        print(f"  BartTorvik Adj DE: {michigan_data['barttorvik']['Adj DE']}")
    
    print("\nCanonical datasets ready for modeling!")
    print(f"  kenpom_canonical.csv: {len(kp)} teams")
    print(f"  barttorvik_canonical.csv: {len(bt)} teams")


if __name__ == '__main__':
    main()
