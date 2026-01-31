"""
Pre-compute bracket simulations and save to JSON files.

This script runs Monte Carlo tournament simulations and saves the results
so the Streamlit app can load them instantly instead of re-computing.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bracket_simulation import (
    load_real_tournament_bracket,
    create_bracket_from_data,
    create_predictor_from_models
)
from data_tools.efficiency_loader import EfficiencyDataLoader


def precompute_bracket(year: int, num_simulations: int) -> dict:
    """Pre-compute bracket simulation for a given year."""
    print(f"Loading {year} tournament bracket...")
    
    # Load tournament bracket
    bracket_data = load_real_tournament_bracket(year)
    
    # Load efficiency data for predictions
    print("Loading efficiency ratings...")
    efficiency_loader = EfficiencyDataLoader()
    kenpom_df = efficiency_loader.load_kenpom()
    
    # Create bracket state and simulator
    print("Creating bracket simulator...")
    bracket_state, simulator = create_bracket_from_data(bracket_data)
    
    # Create game predictor
    game_predictor = create_predictor_from_models(efficiency_data=kenpom_df)
    simulator.game_predictor = game_predictor
    
    # Run simulations
    print(f"Running {num_simulations:,} simulations...")
    simulation_results = simulator.simulate_bracket(bracket_state, num_simulations=num_simulations)
    
    return {
        'year': year,
        'num_simulations': num_simulations,
        'computed_at': datetime.now().isoformat(),
        'bracket_data': bracket_data,
        'simulation_results': serialize_results(simulation_results)
    }


def serialize_results(sim_results: dict) -> dict:
    """Convert simulation results to JSON-serializable format."""
    serialized = {}
    
    for team_id, stats in sim_results.items():
        team = stats['team']
        serialized[team_id] = {
            'team': {
                'name': team.name,
                'seed': team.seed,
                'region': team.region
            },
            'round_32_prob': stats.get('round_32_prob', 0.0),
            'sweet_16_prob': stats.get('sweet_16_prob', 0.0),
            'elite_8_prob': stats.get('elite_8_prob', 0.0),
            'final_four_prob': stats.get('final_four_prob', 0.0),
            'championship_prob': stats.get('championship_prob', 0.0),
            'winner_prob': stats.get('winner_prob', 0.0)
        }
    
    return serialized


def main():
    parser = argparse.ArgumentParser(description='Pre-compute tournament bracket simulations')
    parser.add_argument(
        '--simulations',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations to run (default: 10000)'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=[2025, 2024, 2023],
        help='Tournament years to pre-compute (default: 2025 2024 2023)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path('data_files/precomputed_brackets')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-compute for each year
    for year in args.years:
        try:
            print(f"\n{'='*60}")
            print(f"Pre-computing bracket for {year}")
            print(f"{'='*60}\n")
            
            results = precompute_bracket(year, args.simulations)
            
            # Save to JSON
            output_file = output_dir / f'bracket_{year}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n[OK] Saved results to {output_file}")
            print(f"  - {len(results['simulation_results'])} teams")
            print(f"  - {results['num_simulations']:,} simulations")
            
        except Exception as e:
            print(f"\n[ERROR] Error computing bracket for {year}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
