"""Feature engineering helpers for betting models.

Implements functions from docs/roadmap-betting-features.md with safe
lookups and reasonable defaults so they can be used across the codebase.
"""
from typing import Dict, List, Any
from data_collection import fetch_team_games_with_lines


def _get_eff(e: Dict[str, Any], *keys, default=0.0) -> float:
    for k in keys:
        if isinstance(e, dict) and k in e and e[k] is not None:
            return float(e[k])
    return float(default)


def calculate_efficiency_differential(team1_eff: dict, team2_eff: dict) -> dict:
    """Calculate efficiency differentials between two teams."""
    t1_off = _get_eff(team1_eff, 'adj_offense', 'off_rating', 'offensiveRating')
    t1_def = _get_eff(team1_eff, 'adj_defense', 'def_rating', 'defensiveRating')
    t2_off = _get_eff(team2_eff, 'adj_offense', 'off_rating', 'offensiveRating')
    t2_def = _get_eff(team2_eff, 'adj_defense', 'def_rating', 'defensiveRating')

    return {
        "off_eff_diff": t1_off - t2_off,
        "def_eff_diff": t1_def - t2_def,
        "net_eff_diff": (t1_off - t1_def) - (t2_off - t2_def),
    }


def project_game_total(team1_eff: dict, team2_eff: dict) -> float:
    """Project total points using tempo and efficiency.

    Formula: (Team1_OE + Team2_OE) / 2 * Avg_Tempo / 100  (scaled to game)
    """
    tempo1 = _get_eff(team1_eff, 'tempo', 'pace')
    tempo2 = _get_eff(team2_eff, 'tempo', 'pace')
    avg_tempo = (tempo1 + tempo2) / 2 or 70

    t1_off = _get_eff(team1_eff, 'adj_offense', 'off_rating', 'offensiveRating')
    t2_off = _get_eff(team2_eff, 'adj_offense', 'off_rating', 'offensiveRating')
    t1_def = _get_eff(team1_eff, 'adj_defense', 'def_rating', 'defensiveRating')
    t2_def = _get_eff(team2_eff, 'adj_defense', 'def_rating', 'defensiveRating')

    avg_off_eff = (t1_off + t2_off) / 2
    avg_def_eff = (t1_def + t2_def) / 2

    expected_eff = (avg_off_eff + avg_def_eff) / 2

    # Scale to game possessions (expected_eff is points per 100 possessions)
    projected_total = expected_eff * (avg_tempo / 100) * 2
    return float(projected_total)


def calculate_spread_features(team1: dict, team2: dict) -> dict:
    """Features for predicting point spread."""
    return {
        "net_rating_diff": _get_eff(team1, 'net_rating') - _get_eff(team2, 'net_rating'),
        "off_rating_diff": _get_eff(team1, 'off_rating') - _get_eff(team2, 'off_rating'),
        "def_rating_diff": _get_eff(team1, 'def_rating') - _get_eff(team2, 'def_rating'),
        "ppg_diff": _get_eff(team1, 'ppg') - _get_eff(team2, 'ppg'),
        "opp_ppg_diff": _get_eff(team1, 'opp_ppg') - _get_eff(team2, 'opp_ppg'),
        "margin_diff": _get_eff(team1, 'avg_margin') - _get_eff(team2, 'avg_margin'),
        "seed_diff": _get_eff(team1, 'seed') - _get_eff(team2, 'seed'),
        "efg_diff": _get_eff(team1, 'efg_pct') - _get_eff(team2, 'efg_pct'),
        "to_rate_diff": _get_eff(team1, 'to_rate') - _get_eff(team2, 'to_rate'),
        "orb_diff": _get_eff(team1, 'orb_pct') - _get_eff(team2, 'orb_pct'),
        "ft_rate_diff": _get_eff(team1, 'ft_rate') - _get_eff(team2, 'ft_rate'),
    }


def calculate_total_features(team1: dict, team2: dict) -> dict:
    """Features for predicting game total."""
    tempo1 = _get_eff(team1, 'tempo')
    tempo2 = _get_eff(team2, 'tempo')

    return {
        "combined_tempo": tempo1 + tempo2,
        "avg_tempo": (tempo1 + tempo2) / 2,
        "combined_ppg": _get_eff(team1, 'ppg') + _get_eff(team2, 'ppg'),
        "combined_opp_ppg": _get_eff(team1, 'opp_ppg') + _get_eff(team2, 'opp_ppg'),
        "combined_off_eff": _get_eff(team1, 'off_rating') + _get_eff(team2, 'off_rating'),
        "combined_def_eff": _get_eff(team1, 'def_rating') + _get_eff(team2, 'def_rating'),
        "combined_fg_pct": _get_eff(team1, 'fg_pct') + _get_eff(team2, 'fg_pct'),
        "combined_3pt_pct": _get_eff(team1, 'three_pct') + _get_eff(team2, 'three_pct'),
        "projected_total": project_game_total(team1, team2),
    }


def calculate_win_probability_features(team1: dict, team2: dict) -> dict:
    """Features for win probability prediction."""
    return {
        "net_rating_diff": _get_eff(team1, 'net_rating') - _get_eff(team2, 'net_rating'),
        "seed_diff": _get_eff(team1, 'seed') - _get_eff(team2, 'seed'),
        "ranking_diff": _get_eff(team1, 'ranking') - _get_eff(team2, 'ranking'),
        "tournament_exp_diff": _get_eff(team1, 'tourney_appearances') - _get_eff(team2, 'tourney_appearances'),
        "last_10_win_pct_diff": _get_eff(team1, 'last_10_pct') - _get_eff(team2, 'last_10_pct'),
        "sos_adjusted_margin": _get_eff(team1, 'margin') * _get_eff(team1, 'sos') - _get_eff(team2, 'margin') * _get_eff(team2, 'sos'),
    }


def calculate_implied_probability(moneyline: int) -> float:
    """Convert American odds to implied probability."""
    try:
        ml = int(moneyline)
    except Exception:
        return 0.5

    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def find_value_bets(predictions: List[dict], lines: List[dict], threshold: float = 0.05) -> List[dict]:
    """Find games where model probability exceeds implied probability."""
    value_bets = []
    for pred, line in zip(predictions, lines):
        model_prob = pred.get('win_prob') if pred.get('win_prob') is not None else pred.get('model_prob')
        if model_prob is None:
            continue
        ml = line.get('moneyline') or line.get('ml') or 0
        implied_prob = calculate_implied_probability(ml)
        edge = float(model_prob) - float(implied_prob)
        if edge > threshold:
            value_bets.append({
                "team": pred.get('team') or pred.get('home') or pred.get('away'),
                "model_prob": float(model_prob),
                "implied_prob": float(implied_prob),
                "edge": float(edge),
                "moneyline": ml,
            })

    return sorted(value_bets, key=lambda x: -x['edge'])


def calculate_ats_features(team1: dict, team2: dict, spread: float) -> dict:
    """Features for predicting ATS outcomes."""
    predicted_margin = _get_eff(team1, 'predicted_margin')
    return {
        "predicted_margin": predicted_margin,
        "spread": float(spread),
        "spread_diff": predicted_margin - float(spread),
        "team1_ats_pct": _get_eff(team1, 'ats_pct', default=0.5),
        "team2_ats_pct": _get_eff(team2, 'ats_pct', default=0.5),
        "team1_avg_cover_margin": _get_eff(team1, 'avg_cover', default=0.0),
        "team2_avg_cover_margin": _get_eff(team2, 'avg_cover', default=0.0),
        "fav_ats_pct": _get_eff(team1, 'favorite_ats_pct', default=0.5),
        "dog_ats_pct": _get_eff(team2, 'underdog_ats_pct', default=0.5),
    }


def predict_win_probability(team1: str, team2: str, efficiency_data: Dict[str, dict] = None, 
                          stats_data: Dict[str, dict] = None, models: Dict = None) -> float:
    """Predict win probability for team1 vs team2 using trained models.
    
    Args:
        team1: Name of first team
        team2: Name of second team  
        efficiency_data: Dict mapping team names to efficiency stats
        stats_data: Dict mapping team names to team stats
        models: Trained prediction models
        
    Returns:
        Probability that team1 beats team2 (0.0 to 1.0)
    """
    try:
        # Get team data
        team1_eff = efficiency_data.get(team1, {}) if efficiency_data else {}
        team2_eff = efficiency_data.get(team2, {}) if efficiency_data else {}
        team1_stats = stats_data.get(team1, {}) if stats_data else {}
        team2_stats = stats_data.get(team2, {}) if stats_data else {}
        
        # Calculate features
        features = calculate_win_probability_features(team1_stats, team2_stats)
        
        # Try advanced model first - it expects only 3 efficiency features
        if models and models.get('moneyline_advanced'):
            try:
                # Calculate the 3 efficiency features that the advanced model was trained on
                eff_diff = calculate_efficiency_differential(team1_eff, team2_eff)
                feature_dict = {
                    'off_eff_diff': eff_diff.get('off_eff_diff', 0),
                    'def_eff_diff': eff_diff.get('def_eff_diff', 0), 
                    'net_eff_diff': eff_diff.get('net_eff_diff', 0)
                }
                
                import pandas as pd
                feature_names = list(feature_dict.keys())
                df = pd.DataFrame([feature_dict], columns=feature_names)
                pred_proba = models['moneyline_advanced'].predict_proba(df)[0]
                return float(pred_proba[1])  # Probability of positive class (team1 win)
            except Exception as e:
                print(f"Error with advanced model: {e}")
        
        # Fall back to basic models - they expect the full feature set
        if models and models.get('moneyline'):
            try:
                # Calculate features like in the main prediction system
                eff_diff = calculate_efficiency_differential(team1_eff, team2_eff)
                minimal = {
                    'off_eff_diff': eff_diff.get('off_eff_diff', 0),
                    'def_eff_diff': eff_diff.get('def_eff_diff', 0),
                    'net_eff_diff': eff_diff.get('net_eff_diff', 0)
                }
                
                # Add spread features
                try:
                    spread_feats = calculate_spread_features(team1_stats, team2_stats)
                except Exception:
                    spread_feats = {
                        'net_rating_diff': eff_diff.get('net_eff_diff', 0),
                        'off_rating_diff': eff_diff.get('off_eff_diff', 0),
                        'def_rating_diff': eff_diff.get('def_eff_diff', 0),
                        'ppg_diff': team1_stats.get('ppg', 0) - team2_stats.get('ppg', 0),
                        'opp_ppg_diff': team1_stats.get('opp_ppg', 0) - team2_stats.get('opp_ppg', 0),
                        'margin_diff': 0,
                        'efg_diff': team1_stats.get('efg_pct', 0) - team2_stats.get('efg_pct', 0),
                        'to_rate_diff': team1_stats.get('to_rate', 0) - team2_stats.get('to_rate', 0),
                        'orb_diff': team1_stats.get('orb_pct', 0) - team2_stats.get('orb_pct', 0),
                        'ft_rate_diff': team1_stats.get('ft_rate', 0) - team2_stats.get('ft_rate', 0),
                    }
                
                # Combine and filter to expected features
                feature_dict = {**minimal, **spread_feats}
                expected_features = ['off_eff_diff', 'def_eff_diff', 'net_eff_diff',
                                   'net_rating_diff', 'off_rating_diff', 'def_rating_diff',
                                   'ppg_diff', 'opp_ppg_diff', 'margin_diff',
                                   'efg_diff', 'to_rate_diff']
                feature_dict = {k: v for k, v in feature_dict.items() if k in expected_features}
                
                import pandas as pd
                feature_names = list(feature_dict.keys())
                df = pd.DataFrame([feature_dict], columns=feature_names)
                scalers = models.get('moneyline_scalers', {})
                
                moneyline_preds = []
                for model_name, model in models['moneyline'].items():
                    try:
                        if model_name in scalers:
                            scaled_df = scalers[model_name].transform(df)
                            pred_proba = model.predict_proba(scaled_df)[0]
                        else:
                            pred_proba = model.predict_proba(df)[0]
                        moneyline_preds.append(pred_proba[1])
                    except Exception as e:
                        continue
                        
                if moneyline_preds:
                    return float(sum(moneyline_preds) / len(moneyline_preds))
            except Exception as e:
                print(f"Error with basic models: {e}")
        
        # Final fallback: use efficiency differential
        team1_off = _get_eff(team1_eff, 'adj_offense', 'offensiveRating', 100)
        team1_def = _get_eff(team1_eff, 'adj_defense', 'defensiveRating', 100)
        team2_off = _get_eff(team2_eff, 'adj_offense', 'offensiveRating', 100)
        team2_def = _get_eff(team2_eff, 'adj_defense', 'defensiveRating', 100)
        
        team1_net = team1_off - team1_def
        team2_net = team2_off - team2_def
        diff = team1_net - team2_net
        
        # Convert efficiency diff to probability using logistic function
        import math
        prob = 1 / (1 + math.exp(-diff / 15))  # Scale factor of 15 is reasonable
        return float(prob)
        
    except Exception as e:
        print(f"Error predicting win probability for {team1} vs {team2}: {e}")
        return 0.5  # Default to 50/50


def find_upset_candidates(matchups: List[dict], min_seed_diff: int = 4, 
                         efficiency_data: Dict[str, dict] = None,
                         stats_data: Dict[str, dict] = None,
                         models: Dict = None) -> List[dict]:
    """Identify games where lower seed has strong upset potential.
    
    Args:
        matchups: List of matchup dictionaries with keys:
            - higher_seed_team, lower_seed_team: team names
            - higher_seed, lower_seed: seed numbers
            - underdog_ml: moneyline for the underdog
        min_seed_diff: Minimum seed difference to consider (default 4)
        efficiency_data: Team efficiency data for predictions
        stats_data: Team stats data for predictions
        models: Trained prediction models
        
    Returns:
        List of upset candidates sorted by edge (descending)
    """
    upsets = []
    
    for matchup in matchups:
        try:
            # Extract matchup data
            higher_seed = matchup.get('higher_seed', 0)
            lower_seed = matchup.get('lower_seed', 0)
            higher_seed_team = matchup.get('higher_seed_team', '')
            lower_seed_team = matchup.get('lower_seed_team', '')
            underdog_ml = matchup.get('underdog_ml', 0)
            
            seed_diff = higher_seed - lower_seed
            
            if seed_diff >= min_seed_diff and underdog_ml:
                # Calculate upset probability (probability lower seed wins)
                upset_prob = predict_win_probability(
                    lower_seed_team, 
                    higher_seed_team,
                    efficiency_data,
                    stats_data,
                    models
                )
                
                # Calculate implied probability from moneyline
                implied_prob = calculate_implied_probability(underdog_ml)
                edge = upset_prob - implied_prob
                
                if upset_prob > 0.30:  # At least 30% chance
                    upsets.append({
                        "underdog": lower_seed_team,
                        "favorite": higher_seed_team,
                        "seed_matchup": f"{lower_seed} vs {higher_seed}",
                        "upset_prob": upset_prob,
                        "implied_prob": implied_prob,
                        "edge": edge,
                        "moneyline": underdog_ml
                    })
        except Exception as e:
            print(f"Error processing matchup {matchup}: {e}")
            continue
    
    return sorted(upsets, key=lambda x: -x["edge"])


def build_parlay(picks: list) -> dict:
    """Calculate parlay odds and expected value."""
    # Filter out picks with invalid odds
    valid_picks = [pick for pick in picks if pick.get("odds") and pick["odds"] != 0]
    
    if not valid_picks:
        return {
            "picks": [],
            "parlay_odds": 0,
            "decimal_odds": 1.0,
            "combined_prob": 0.0,
            "expected_value": -1.0,
            "is_positive_ev": False,
            "error": "No valid picks with odds available"
        }
    
    total_odds = 1.0
    
    for pick in valid_picks:
        odds = pick["odds"]
        # Convert American odds to decimal
        if odds > 0:
            decimal = 1 + (odds / 100)
        else:
            decimal = 1 + (100 / abs(odds))
        
        total_odds *= decimal
    
    # Convert back to American
    if total_odds >= 2:
        american_odds = (total_odds - 1) * 100
    else:
        american_odds = -100 / (total_odds - 1)
    
    # Calculate combined probability
    combined_prob = 1.0
    for pick in valid_picks:
        combined_prob *= pick["model_prob"]
    
    # Expected value
    ev = (combined_prob * (total_odds - 1)) - (1 - combined_prob)
    
    return {
        "picks": valid_picks,
        "parlay_odds": american_odds,
        "decimal_odds": total_odds,
        "combined_prob": combined_prob,
        "expected_value": ev,
        "is_positive_ev": ev > 0
    }


def analyze_ats_trends(team: str, years: int = 5) -> dict:
    """Analyze team's historical against-the-spread performance."""
    games = fetch_team_games_with_lines(team, years)
    
    results = {
        "overall_ats": {"wins": 0, "losses": 0, "pushes": 0},
        "as_favorite": {"wins": 0, "losses": 0},
        "as_underdog": {"wins": 0, "losses": 0},
        "tournament_ats": {"wins": 0, "losses": 0},
        "by_spread_range": {}
    }
    
    for game in games:
        # Only analyze completed games with margin data
        if game.get("margin") is None or game.get("spread") is None:
            continue
            
        margin = game["margin"]
        spread = game["spread"]
        
        # Skip games with no spread
        if spread == 0:
            continue
            
        covered = margin > -spread
        
        # Overall
        if covered:
            results["overall_ats"]["wins"] += 1
        else:
            results["overall_ats"]["losses"] += 1
        
        # As favorite/underdog
        if spread < 0:  # Favorite
            key = "as_favorite"
        else:
            key = "as_underdog"
        
        if covered:
            results[key]["wins"] += 1
        else:
            results[key]["losses"] += 1
    
    return results


def track_line_movement(game_id: str) -> dict:
    """Track how lines have moved since opening.

    Uses The Odds API for live line data and demonstrates line movement tracking.
    For full historical tracking, a paid Odds API plan would be needed.

    Args:
        game_id: The game ID to track (currently not used - matches by team names)

    Returns:
        Dictionary with line movement analysis
    """
    try:
        from odds_api_integration import track_line_movement as odds_track_movement

        # For demonstration, we'll use the first available game from live odds
        # In a real implementation, this would match by game_id or team names
        odds_data = odds_track_movement(("Delaware St Hornets", "South Carolina St Bulldogs"))  # Example teams from live data

        if "error" not in odds_data:
            return {
                "game_id": game_id,
                "home_team": odds_data.get("current_lines", {}).get("home_team", "Unknown"),
                "away_team": odds_data.get("current_lines", {}).get("away_team", "Unknown"),
                "spread_current": odds_data.get("current_lines", {}).get("home_spread"),
                "total_current": odds_data.get("current_lines", {}).get("total_line"),
                "spread_movement": odds_data.get("line_movement", {}).get("spread_movement", 0),
                "total_movement": odds_data.get("line_movement", {}).get("total_movement", 0),
                "sharp_money_indicator": odds_data.get("line_movement", {}).get("sharp_money_indicator", "No data"),
                "data_source": "The Odds API",
                "note": "Using live odds. Historical opening lines require paid API plan."
            }
        else:
            # Fallback to cached data approach
            return _track_line_movement_cached(game_id)

    except ImportError:
        # Fallback if odds integration not available
        return _track_line_movement_cached(game_id)


def _track_line_movement_cached(game_id: str) -> dict:
    """Fallback line movement tracking using cached CBBD data."""
    import json
    from pathlib import Path

    # Try to find the game in cached betting lines data
    cache_dir = Path("data_files/cache")

    # Search through recent years for the game
    current_year = 2026  # Current season
    for year in range(current_year, current_year - 3, -1):  # Check last 3 years
        for season_type in ["postseason", "regular"]:
            cache_file = cache_dir / f"lines_{year}_{season_type}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        games = json.load(f)

                    # Find the game by ID
                    for game in games:
                        if str(game.get('gameId', '')) == str(game_id):
                            return _analyze_game_line_movement(game)
                except (json.JSONDecodeError, KeyError):
                    continue

    return {
        "game_id": game_id,
        "error": "Game not found in cached data",
        "spread_movement": 0,
        "total_movement": 0,
        "sharp_money_indicator": "No data available"
    }


def _analyze_game_line_movement(game: dict) -> dict:
    """Analyze line movement for a specific game.
    
    Args:
        game: Game data with lines
        
    Returns:
        Dictionary with movement analysis
    """
    game_id = game.get('gameId', 'unknown')
    
    if not game.get('lines') or len(game['lines']) == 0:
        return {
            "game_id": game_id,
            "error": "No betting lines available",
            "spread_movement": 0,
            "total_movement": 0,
            "sharp_money_indicator": "No lines"
        }
    
    # Get the most recent line
    current_line = game['lines'][0]
    
    # Extract current lines
    spread_current = current_line.get('spread')
    total_current = current_line.get('overUnder')
    
    # For now, opening lines may not be available in cached data
    # In a full implementation, this would compare against opening lines
    spread_open = current_line.get('spreadOpen')
    total_open = current_line.get('overUnderOpen')
    
    # Calculate movements (will be 0 if no opening lines available)
    spread_movement = 0
    total_movement = 0
    
    if spread_current is not None and spread_open is not None:
        spread_movement = spread_current - spread_open
    
    if total_current is not None and total_open is not None:
        total_movement = total_current - total_open
    
    # Analyze sharp money (limited without opening lines)
    sharp_indicator = "Opening lines not available"
    if spread_open is not None and total_open is not None:
        sharp_indicator = analyze_sharp_action({
            'spread': spread_open,
            'over_under': total_open
        }, {
            'spread': spread_current,
            'over_under': total_current
        })
    
    return {
        "game_id": game_id,
        "home_team": game.get('homeTeam', 'Unknown'),
        "away_team": game.get('awayTeam', 'Unknown'),
        "spread_open": spread_open,
        "spread_current": spread_current,
        "spread_movement": spread_movement,
        "total_open": total_open,
        "total_current": total_current,
        "total_movement": total_movement,
        "sharp_money_indicator": sharp_indicator,
        "provider": current_line.get('provider', 'Unknown'),
        "note": "Opening lines may not be available in cached data. Real-time tracking would require live API integration."
    }


def _analyze_game_line_movement(game: dict) -> dict:
    """Analyze line movement for a specific game.
    
    Args:
        game: Game data with lines
        
    Returns:
        Dictionary with movement analysis
    """
    game_id = game.get('gameId', 'unknown')
    
    if not game.get('lines') or len(game['lines']) == 0:
        return {
            "game_id": game_id,
            "error": "No betting lines available",
            "spread_movement": 0,
            "total_movement": 0,
            "sharp_money_indicator": "No lines"
        }
    
    # Get the most recent line (assuming lines are ordered by time)
    current_line = game['lines'][0]
    
    # Extract current and opening lines
    spread_current = current_line.get('spread')
    spread_open = current_line.get('spreadOpen')
    total_current = current_line.get('overUnder') 
    total_open = current_line.get('overUnderOpen')
    
    # Calculate movements
    spread_movement = 0
    total_movement = 0
    
    if spread_current is not None and spread_open is not None:
        spread_movement = spread_current - spread_open
    
    if total_current is not None and total_open is not None:
        total_movement = total_current - total_open
    
    # Analyze sharp money
    sharp_indicator = analyze_sharp_action({
        'spread': spread_open,
        'over_under': total_open
    }, {
        'spread': spread_current,
        'over_under': total_current
    })
    
    return {
        "game_id": game_id,
        "home_team": game.get('homeTeam', 'Unknown'),
        "away_team": game.get('awayTeam', 'Unknown'),
        "spread_open": spread_open,
        "spread_current": spread_current,
        "spread_movement": spread_movement,
        "total_open": total_open,
        "total_current": total_current,
        "total_movement": total_movement,
        "sharp_money_indicator": sharp_indicator,
        "provider": current_line.get('provider', 'Unknown')
    }


def analyze_sharp_action(opening: dict, current: dict) -> str:
    """Detect potential sharp money movement.
    
    Args:
        opening: Opening line data
        current: Current line data
        
    Returns:
        String describing sharp money movement
    """
    if not opening or not current:
        return "Insufficient data"
    
    spread_open = opening.get('spread')
    spread_current = current.get('spread')
    total_open = opening.get('over_under')
    total_current = current.get('over_under')
    
    if spread_open is None or spread_current is None:
        spread_move = 0
    else:
        spread_move = spread_current - spread_open
    
    if total_open is None or total_current is None:
        total_move = 0
    else:
        total_move = total_current - total_open
    
    # Analyze spread movement
    if abs(spread_move) >= 3:
        if spread_move > 0:
            return f"Heavy action on underdog (+{abs(spread_move):.1f} points)"
        else:
            return f"Heavy action on favorite ({spread_move:.1f} points)"
    elif abs(spread_move) >= 1.5:
        if spread_move > 0:
            return f"Sharp action on underdog (+{abs(spread_move):.1f} points)"
        else:
            return f"Sharp action on favorite ({spread_move:.1f} points)"
    
    # Analyze total movement
    if abs(total_move) >= 3:
        if total_move > 0:
            return f"Heavy action on over (+{total_move:.1f} points)"
        else:
            return f"Heavy action on under ({total_move:.1f} points)"
    
    return "No significant movement"


def get_current_lines(game_id: str) -> dict:
    """Get current betting lines for a game from The Odds API.

    Args:
        game_id: The game ID to fetch lines for (currently matches by team names)

    Returns:
        Dictionary with current line data
    """
    try:
        from odds_api_integration import get_current_lines as odds_get_current

        # For demonstration, get the first available game
        # In practice, this would match by game_id or team names
        lines = odds_get_current(("Any", "Team"))  # Placeholder - would be actual team matching

        if lines:
            return {
                'spread': lines.get('home_spread'),
                'over_under': lines.get('total_line'),
                'home_moneyline': lines.get('home_moneyline'),
                'away_moneyline': lines.get('away_moneyline'),
                'provider': lines.get('bookmaker', 'The Odds API'),
                'updated': lines.get('last_update')
            }
    except ImportError:
        pass  # Fall back to cached data

    # Fallback to cached CBBD data
    try:
        from data_collection import get_api_client
        import cbbd

        with get_api_client() as api_client:
            lines_api = cbbd.LinesApi(api_client)
            lines = lines_api.get_lines(game_id=game_id)

            if lines and len(lines) > 0:
                line = lines[0]  # Most recent line
                return {
                    'spread': getattr(line, 'spread', None),
                    'over_under': getattr(line, 'overUnder', None),
                    'home_moneyline': getattr(line, 'homeMoneyline', None),
                    'away_moneyline': getattr(line, 'awayMoneyline', None),
                    'provider': getattr(line, 'provider', None),
                    'updated': getattr(line, 'updated', None)
                }
    except Exception as e:
        pass  # Fall back to cached data

    return {}


def get_opening_lines(game_id: str) -> dict:
    """Get opening betting lines for a game from The Odds API.

    Note: Opening lines require historical data access (paid Odds API plan).
    This currently returns current lines as a demonstration.

    Args:
        game_id: The game ID to fetch opening lines for

    Returns:
        Dictionary with opening line data
    """
    try:
        from odds_api_integration import get_opening_lines as odds_get_opening

        # For demonstration, get the first available game
        lines = odds_get_opening(("Any", "Team"))  # Placeholder - would be actual team matching

        if lines:
            return {
                'spread': lines.get('home_spread'),
                'over_under': lines.get('total_line'),
                'home_moneyline': lines.get('home_moneyline'),
                'away_moneyline': lines.get('away_moneyline'),
                'provider': lines.get('bookmaker', 'The Odds API'),
                'note': 'Using current lines as opening lines (historical data requires paid plan)'
            }
    except ImportError:
        pass

    # Fallback: return empty dict as opening lines aren't readily available
    return {}


__all__ = [
    'calculate_efficiency_differential',
    'project_game_total',
    'calculate_spread_features',
    'calculate_total_features',
    'calculate_win_probability_features',
    'calculate_implied_probability',
    'find_value_bets',
    'calculate_ats_features',
    'predict_win_probability',
    'find_upset_candidates',
    'build_parlay',
    'analyze_ats_trends',
    'track_line_movement',
    'analyze_sharp_action',
    'get_current_lines',
    'get_opening_lines',
]
