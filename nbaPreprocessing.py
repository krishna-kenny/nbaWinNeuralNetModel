import pandas as pd

# Load each CSV file and drop rows with null values

# File 1: Player Data
playerFile = 'nba_player_info.csv'
nba_player_info = pd.read_csv(playerFile)
nba_player_info_cleaned = nba_player_info.dropna()

# File 2: Team Data
teamFile = 'nba_team_info.csv'
nba_team_info = pd.read_csv(teamFile)
nba_team_info_cleaned = nba_team_info.dropna()

# File 3: Game Data
gameFile = 'nba_game_logs.csv'
nba_game_logs = pd.read_csv(gameFile)
nba_game_logs_cleaned = nba_game_logs.dropna()

# The cleaned data is now stored in nba_player_info_cleaned, nba_team_info_cleaned, and nba_game_logs_cleaned
