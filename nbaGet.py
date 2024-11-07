import pandas as pd
from nba_api.stats.endpoints import TeamInfoCommon, CommonAllPlayers, TeamGameLogs
from nba_api.stats.static import teams
import time

def fetch_team_ids():
    nba_teams = teams.get_teams()
    team_ids_dict = {team['abbreviation']: team['id'] for team in nba_teams}
    return team_ids_dict


def get_team_info(seasons):
    nba_teams = teams.get_teams()
    team_data = []

    for season in seasons:
        for team in nba_teams:
            try:
                team_info = TeamInfoCommon(team_id=team['id'], season_type_nullable='Regular Season')
                df_team = team_info.get_data_frames()[0]
                df_team['TeamName'] = team['full_name']
                df_team['Season'] = season
                team_data.append(df_team)
                time.sleep(0.6)  # Add delay to avoid API rate limit
            except Exception as e:
                print(f"Error fetching data for team {team['full_name']} in season {season}: {e}")
                continue  # Skip team if there's an error

    if team_data:
        df_teams = pd.concat(team_data, ignore_index=True)
        df_teams.to_csv("data/nba_team_info.csv", index=False)
    else:
        print("No team data fetched.")

def get_player_info(seasons):
    player_data = []

    for season in seasons:
        try:
            players = CommonAllPlayers(is_only_current_season=1, season=season)
            df_players = players.get_data_frames()[0]
            df_players['Season'] = season
            player_data.append(df_players)
            time.sleep(0.6)
        except Exception as e:
            print(f"Error fetching player info for season {season}: {e}")
            continue  # Skip season if there's an error

    if player_data:
        df_all_players = pd.concat(player_data, ignore_index=True)
        df_all_players.to_csv("data/nba_player_info.csv", index=False)
    else:
        print("No player data fetched.")

def get_game_logs(seasons):
    game_log_data = []

    for season in seasons:
        try:
            game_logs = TeamGameLogs(season_nullable=season, season_type_nullable='Regular Season')
            df_game_logs = game_logs.get_data_frames()[0]
            df_game_logs['Season'] = season
            game_log_data.append(df_game_logs)
            time.sleep(0.6)
        except Exception as e:
            print(f"Error fetching game logs for season {season}: {e}")
            continue  # Skip season if there's an error

    if game_log_data:
        df_all_game_logs = pd.concat(game_log_data, ignore_index=True)
        df_all_game_logs.to_csv("data/nba_game_logs.csv", index=False)
    else:
        print("No game log data fetched.")

# Define the list of seasons
seasons = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
currentSeason = ['2024-25']

# Run functions to save data to CSV files
get_team_info(currentSeason)
print("get_team_info stored.")
get_player_info(currentSeason)
print("get_player_info stored.")
get_game_logs(seasons)
print("get_game_logs stored.")

