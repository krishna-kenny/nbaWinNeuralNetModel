import pandas as pd
import numpy as np
from nba_api.stats.endpoints import TeamGameLogs
from nba_api.stats.static import teams
from nbaModel import load_trained_model  # Ensure this module is correctly referenced

# Get Team ID based on Team Name
def get_team_id_by_name(team_name):
    """Retrieve the team ID by team name."""
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if team['full_name'].lower() == team_name.lower():
            return team['id']
    raise ValueError(f"Team '{team_name}' not found!")

# Fetch and process latest game stats (last 16 games) for a team
def fetch_last_16_games_stats(team_id):
    """Fetch and compute stats for the last 16 games of a team."""
    team_game_logs = TeamGameLogs(
        team_id_nullable=team_id,
        league_id_nullable='00',
        season_nullable='2023-24',
        season_type_nullable='Regular Season'
    ).get_data_frames()[0].head(16)

    # Calculate win/loss history
    win_loss_history = team_game_logs['WL'].apply(lambda x: 1 if x == 'W' else 0).tolist()

    # Calculate average stats over last 16 games
    avg_stats = team_game_logs[['PTS', 'FGM', 'FGA', 'FG3M', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV']].mean().to_numpy()

    return win_loss_history, avg_stats

# Calculate head-to-head win/loss history (last 8 games)
def fetch_head_to_head_history(team1_id, team2_id):
    """Fetch head-to-head win/loss history between two teams."""
    team1_logs = TeamGameLogs(
        team_id_nullable=team1_id,
        league_id_nullable='00',
        season_nullable='2023-24',
        season_type_nullable='Regular Season'
    ).get_data_frames()[0]

    # Correct MATCHUP string formatting
    matchup_string = f'{team1_id} vs. {team2_id}'
    head_to_head_logs = team1_logs[team1_logs['MATCHUP'].str.contains(matchup_string)].head(8)

    return head_to_head_logs['WL'].apply(lambda x: 1 if x == 'W' else 0).tolist()

# Calculate opponent defensive strength (average over past games)
def calculate_defensive_strength(team_id):
    """Calculate the defensive strength of a team."""
    team_game_logs = TeamGameLogs(
        team_id_nullable=team_id,
        league_id_nullable='00',
        season_nullable='2023-24',
        season_type_nullable='Regular Season'
    ).get_data_frames()[0].head(16)

    # Opponent defensive stats
    opponent_defense = team_game_logs[['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'TOV']].mean().to_numpy()
    return opponent_defense

# Predict win probability
def predict_matchup_win_probability(team1_name, team2_name):
    """Predict the win probability of team1 against team2."""
    model = load_trained_model()
    team1_id = get_team_id_by_name(team1_name)
    team2_id = get_team_id_by_name(team2_name)

    # Fetch stats for each team
    team1_wl_history, team1_avg_stats = fetch_last_16_games_stats(team1_id)
    team2_wl_history, team2_avg_stats = fetch_last_16_games_stats(team2_id)
    head_to_head_history = fetch_head_to_head_history(team1_id, team2_id)
    team1_defense = calculate_defensive_strength(team1_id)
    team2_defense = calculate_defensive_strength(team2_id)

    # Combine features into a single input array
    features = np.concatenate([
        [team1_id, team2_id],
        team1_wl_history,
        team2_wl_history,
        head_to_head_history,
        team1_avg_stats,
        team2_avg_stats,
        team1_defense,
        team2_defense
    ])

    # Reshape features for model prediction
    features = features.reshape(1, -1)
    
    # Predict win probability for Team 1
    win_probability = model.predict(features)[0][0]
    print(f"Predicted probability of {team1_name} winning against {team2_name}: {win_probability * 100:.2f}%")

def main():
    team1_name = input("Enter Team 1 name: ")
    team2_name = input("Enter Team 2 name: ")
    
    try:
        predict_matchup_win_probability(team1_name, team2_name)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()


