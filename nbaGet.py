import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs, PlayerGameLogs
from nba_api.stats.static import teams, players

def fetch_team_ids():
    """Fetch team IDs dynamically from the NBA API."""
    nba_teams = teams.get_teams()
    return {team['abbreviation']: team['id'] for team in nba_teams}

def fetch_game_logs(team_id, season, num_games=16):
    """Fetch team game logs for a specific team and season, limiting to last num_games."""
    logs = TeamGameLogs(
        team_id_nullable=team_id,
        league_id_nullable='00',
        season_nullable=season,
        season_type_nullable='Regular Season'
    ).get_data_frames()

    if logs and not logs[0].empty:
        return logs[0].head(num_games)
    return pd.DataFrame()  # Return empty DataFrame if no logs

def fetch_team_win_loss_history(team_id, season, num_games=16):
    """Fetch win/loss history for the last num_games games for a team in a season."""
    logs = fetch_game_logs(team_id, season, num_games)
    return logs['WL'].apply(lambda x: 1 if x == 'W' else 0).tolist() if not logs.empty else [0] * num_games

def fetch_head_to_head_history(team1_id, team2_id, season='2023-24', num_games=8):
    """Fetch head-to-head win/loss history between two teams."""
    logs = fetch_game_logs(team1_id, season, num_games)
    head_to_head_logs = logs[logs['MATCHUP'].str.contains(str(team2_id), na=False)]
    return head_to_head_logs['WL'].apply(lambda x: 1 if x == 'W' else 0).tolist()

def get_player_stats(player_id, season):
    """Fetch player statistics for a given player ID for a specific season."""
    logs = PlayerGameLogs(
        player_id=player_id,
        season=season,
        season_type_all_star='Regular Season'
    ).get_data_frames()
    return logs[0] if logs and not logs[0].empty else None

def fetch_player_points_per_minute(team_id, season):
    """Calculate points per minute for each player in a team."""
    team_players = [p for p in players.get_players() if p.get('team_id') == team_id]
    player_ppm = {}
    
    for player in team_players:
        stats = get_player_stats(player['id'], season)
        if stats is not None:
            total_minutes = stats['MIN'].sum()
            if total_minutes > 0:
                player_ppm[player['id']] = stats['PTS'].sum() / total_minutes
    
    return player_ppm

def fetch_avg_team_stats(team_id, season, num_games=16):
    """Fetch average statistics for a team over the last num_games."""
    logs = fetch_game_logs(team_id, season, num_games)
    if logs.empty:
        return {stat: 0 for stat in ['PTS', 'FGM', 'FGA', 'FG3M', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV']}
    
    return {f'AVG_{stat}': logs[stat].mean() for stat in ['PTS', 'FGM', 'FGA', 'FG3M', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV']}

def fetch_opponent_defensive_strength(team_id, season, num_games=16):
    """Fetch defensive stats of opponents over last num_games."""
    logs = fetch_game_logs(team_id, season, num_games)
    if logs.empty:
        return {stat: 0 for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'TOV']}
    
    return {f'DEF_{stat}': logs[stat].mean() for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'TOV']}
