import time
import pandas as pd
from nba_api.stats.endpoints import TeamInfoCommon, TeamGameLogs, PlayerGameLogs
from nba_api.stats.static import teams

# Maximum number of retries for each API call
MAX_RETRIES = 3

def fetch_with_retries(func, *args, **kwargs):
    """Attempts a function call up to MAX_RETRIES with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    print(f"Failed after {MAX_RETRIES} attempts.")
    return None

def get_team_info(seasons):
    print('Fetching team information...')
    nba_teams = teams.get_teams()
    team_data = []

    for season in seasons:
        for team in nba_teams:
            team_info = fetch_with_retries(
                TeamInfoCommon, team_id=team['id'], season_type_nullable='Regular Season', timeout=60
            )
            if team_info:
                df_team = team_info.get_data_frames()[0]
                team_data.append(df_team)
                time.sleep(0.6)  # Delay to avoid API rate limits
            else:
                print(f"Skipping team {team['full_name']} for season {season} after failed attempts.")

    if team_data:
        df_teams = pd.concat(team_data, ignore_index=True)
        df_teams.to_csv("data/raw/nba_team_data.csv", index=False)
    else:
        print("No team data fetched.")

def get_game_logs(seasons):
    print('Fetching game logs...')
    game_log_data = []

    for season in seasons:
        game_logs = fetch_with_retries(TeamGameLogs, season_nullable=season, season_type_nullable='Regular Season', timeout=60)
        if game_logs:
            df_game_logs = game_logs.get_data_frames()[0]
            game_log_data.append(df_game_logs)
            time.sleep(0.6)
        else:
            print(f"Skipping game logs for season {season} after failed attempts.")

    if game_log_data:
        df_all_game_logs = pd.concat(game_log_data, ignore_index=True)
        df_all_game_logs.to_csv("data/raw/nba_game_logs.csv", index=False)
    else:
        print("No game log data fetched.")

def get_player_game_logs(seasons):
    print('Fetching player game logs...')
    player_game_log_data = []

    for season in seasons:
        player_game_logs = fetch_with_retries(PlayerGameLogs, season_nullable=season, season_type_nullable='Regular Season', timeout=60)
        if player_game_logs:
            df_player_game_logs = player_game_logs.get_data_frames()[0]
            player_game_log_data.append(df_player_game_logs)
            time.sleep(0.6)
        else:
            print(f"Skipping player game logs for season {season} after failed attempts.")

    if player_game_log_data:
        df_all_player_game_logs = pd.concat(player_game_log_data, ignore_index=True)
        df_all_player_game_logs.to_csv("data/raw/nba_player_game_logs.csv", index=False)
    else:
        print("No player game log data fetched.")

# Define the list of seasons
seasons = ['2022-23', '2023-24', '2024-25']

# Run functions to save data to CSV files
get_team_info(seasons[-1])
print("Team information data stored.")

get_game_logs(seasons)
print("Game logs data stored.")

get_player_game_logs(seasons[-2:])  # Fetching for the most recent two seasons only
print("Player game logs data stored.")
