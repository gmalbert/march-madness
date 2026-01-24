"""
Quick Reference: Using Efficiency Data Loader

This is a cheat sheet for common operations with KenPom and BartTorvik data.
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

from data_tools.efficiency_loader import EfficiencyDataLoader

# Initialize loader
loader = EfficiencyDataLoader()

# Load canonical datasets (364 teams each)
kenpom_df = loader.load_kenpom()
bart_df = loader.load_barttorvik()

# ============================================================================
# GET DATA FOR A SPECIFIC TEAM
# ============================================================================

# Get KenPom data for one team
duke_kp = loader.get_kenpom_for_team('Duke')
print(duke_kp['NetRtg'])  # 33.25
print(duke_kp['Rk'])      # 3
print(duke_kp['ORtg'])    # 126.2
print(duke_kp['DRtg'])    # 93.0

# Get BartTorvik data for one team
duke_bt = loader.get_barttorvik_for_team('Duke')
print(duke_bt['Adj OE'])  # Adjusted offensive efficiency
print(duke_bt['Adj DE'])  # Adjusted defensive efficiency

# Get merged data from both sources
duke_all = loader.get_merged_efficiency('Duke')
kenpom_data = duke_all['kenpom']
bart_data = duke_all['barttorvik']

# ============================================================================
# FILTER DATAFRAMES
# ============================================================================

# Get top 25 teams by NetRtg
top25 = kenpom_df.nsmallest(25, 'Rk')

# Get all ACC teams
acc_teams = kenpom_df[kenpom_df['Conf'] == 'ACC']

# Get teams with NetRtg > 20
elite_teams = kenpom_df[kenpom_df['NetRtg'] > 20]

# Get specific team
michigan = kenpom_df[kenpom_df['canonical_team'] == 'Michigan']

# ============================================================================
# CALCULATE DIFFERENTIALS FOR A GAME
# ============================================================================

def get_game_differentials(home_team, away_team, kenpom_df):
    """Calculate KenPom differentials for a game."""
    home = kenpom_df[kenpom_df['canonical_team'] == home_team].iloc[0]
    away = kenpom_df[kenpom_df['canonical_team'] == away_team].iloc[0]
    
    return {
        'netrtg_diff': home['NetRtg'] - away['NetRtg'],
        'ortg_diff': home['ORtg'] - away['ORtg'],
        'drtg_diff': home['DRtg'] - away['DRtg'],
        'tempo_diff': home['AdjT'] - away['AdjT'],
        'luck_diff': home['Luck'] - away['Luck'],
        'sos_diff': home['SOS_NetRtg'] - away['SOS_NetRtg']
    }

# Example usage
duke_vs_unc = get_game_differentials('Duke', 'North Carolina', kenpom_df)
print(f"NetRtg advantage: {duke_vs_unc['netrtg_diff']:.2f}")

# ============================================================================
# COMMON KENPOM COLUMNS
# ============================================================================

kenpom_columns = {
    'Rk': 'National rank',
    'Team': 'Original team name from KenPom',
    'canonical_team': 'Standardized ESPN team name',
    'Conf': 'Conference',
    'W-L': 'Win-loss record (string)',
    'wins': 'Wins (int)',
    'losses': 'Losses (int)',
    'win_pct': 'Win percentage (float)',
    'NetRtg': 'Net rating (ORtg - DRtg)',
    'ORtg': 'Offensive rating (points per 100 possessions)',
    'ORtg_Rank': 'National rank for offensive rating',
    'DRtg': 'Defensive rating (points allowed per 100 possessions)',
    'DRtg_Rank': 'National rank for defensive rating',
    'AdjT': 'Adjusted tempo (possessions per 40 minutes)',
    'AdjT_Rank': 'National rank for tempo',
    'Luck': 'Luck rating (deviation from Pythagorean expectation)',
    'Luck_Rank': 'National rank for luck',
    'SOS_NetRtg': 'Strength of schedule (net rating)',
    'SOS_NetRtg_Rank': 'National rank for SOS',
    'SOS_ORtg': 'SOS offensive rating',
    'SOS_DRtg': 'SOS defensive rating',
    'NCSOS_NetRtg': 'Non-conference SOS',
}

# ============================================================================
# COMMON BARTTORVIK COLUMNS
# ============================================================================

bart_columns = {
    'Team': 'Original team name from BartTorvik',
    'canonical_team': 'Standardized ESPN team name',
    'Adj OE': 'Adjusted offensive efficiency',
    'Adj DE': 'Adjusted defensive efficiency',
    # ... see barttorvik_ratings.csv for full column list
}

# ============================================================================
# SAVE CANONICAL DATASETS
# ============================================================================

# This creates/updates kenpom_canonical.csv and barttorvik_canonical.csv
kp, bt = loader.save_canonical_datasets()

# ============================================================================
# ERROR HANDLING
# ============================================================================

# Check if team exists
def safe_get_team_data(team_name, loader):
    """Safely get team data with error handling."""
    kp_data = loader.get_kenpom_for_team(team_name)
    bt_data = loader.get_barttorvik_for_team(team_name)
    
    if kp_data is None:
        print(f"Warning: {team_name} not found in KenPom data")
    
    if bt_data is None:
        print(f"Warning: {team_name} not found in BartTorvik data")
    
    return kp_data, bt_data

# ============================================================================
# COMPARE SOURCES
# ============================================================================

def compare_efficiency_sources(team_name, kenpom_df, bart_df):
    """Compare KenPom and BartTorvik ratings for a team."""
    kp = kenpom_df[kenpom_df['canonical_team'] == team_name]
    bt = bart_df[bart_df['canonical_team'] == team_name]
    
    if kp.empty or bt.empty:
        return None
    
    kp = kp.iloc[0]
    bt = bt.iloc[0]
    
    return {
        'team': team_name,
        'kenpom': {
            'ORtg': kp['ORtg'],
            'DRtg': kp['DRtg'],
            'NetRtg': kp['NetRtg']
        },
        'barttorvik': {
            'Adj OE': bt['Adj OE'],
            'Adj DE': bt['Adj DE'],
            'NetRtg': bt['Adj OE'] - bt['Adj DE']
        },
        'diff': {
            'ORtg': kp['ORtg'] - bt['Adj OE'],
            'DRtg': kp['DRtg'] - bt['Adj DE']
        }
    }

# ============================================================================
# INTEGRATION WITH PREDICTIONS
# ============================================================================

# In predictions.py, this is how data is used:

from data_tools.efficiency_loader import EfficiencyDataLoader

# Load once at startup
loader = EfficiencyDataLoader()
kenpom_df, bart_df = loader.load_kenpom(), loader.load_barttorvik()

# For each game prediction
def enrich_game_with_efficiency(home_team, away_team, kenpom_df, bart_df):
    """Get efficiency metrics for both teams."""
    home_kp = kenpom_df[kenpom_df['canonical_team'] == home_team]
    away_kp = kenpom_df[kenpom_df['canonical_team'] == away_team]
    
    home_bt = bart_df[bart_df['canonical_team'] == home_team]
    away_bt = bart_df[bart_df['canonical_team'] == away_team]
    
    metrics = {
        'home': {},
        'away': {}
    }
    
    if not home_kp.empty:
        metrics['home']['kenpom'] = home_kp.iloc[0].to_dict()
    
    if not away_kp.empty:
        metrics['away']['kenpom'] = away_kp.iloc[0].to_dict()
    
    if not home_bt.empty:
        metrics['home']['barttorvik'] = home_bt.iloc[0].to_dict()
    
    if not away_bt.empty:
        metrics['away']['barttorvik'] = away_bt.iloc[0].to_dict()
    
    return metrics

# ============================================================================
# USEFUL QUERIES
# ============================================================================

# Teams with best offensive efficiency
best_offense = kenpom_df.nsmallest(10, 'ORtg_Rank')[['canonical_team', 'ORtg', 'ORtg_Rank']]

# Teams with best defensive efficiency  
best_defense = kenpom_df.nsmallest(10, 'DRtg_Rank')[['canonical_team', 'DRtg', 'DRtg_Rank']]

# Fastest tempo teams
fastest_tempo = kenpom_df.nlargest(10, 'AdjT')[['canonical_team', 'AdjT', 'AdjT_Rank']]

# Luckiest teams (overperforming)
luckiest = kenpom_df.nlargest(10, 'Luck')[['canonical_team', 'Luck', 'wins', 'losses']]

# Toughest schedule
toughest_sos = kenpom_df.nlargest(10, 'SOS_NetRtg')[['canonical_team', 'SOS_NetRtg', 'Conf']]

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Calculate composite metrics
kenpom_df['net_efficiency'] = kenpom_df['ORtg'] - kenpom_df['DRtg']
kenpom_df['tempo_adjusted_offense'] = kenpom_df['ORtg'] * (kenpom_df['AdjT'] / 70)
kenpom_df['consistency'] = kenpom_df['wins'] / (kenpom_df['wins'] + kenpom_df['losses']) - (kenpom_df['Luck'] + 0.5)

# Normalize ratings to z-scores
from scipy import stats
kenpom_df['netrtg_zscore'] = stats.zscore(kenpom_df['NetRtg'])
kenpom_df['ortg_zscore'] = stats.zscore(kenpom_df['ORtg'])
kenpom_df['drtg_zscore'] = stats.zscore(kenpom_df['DRtg'])
