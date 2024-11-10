import pandas as pd

# Function to preprocess team data
def preprocess_team_data(input_file, output_file):
    """Load, clean, and save team data by removing unnecessary columns."""
    team_data = pd.read_csv(input_file).dropna()
    columns_to_drop = [
        'TEAM_NAME', 'TEAM_CITY', 'SEASON_YEAR', 
        'TEAM_CODE', 'TEAM_DIVISION', 'MIN_YEAR', 'MAX_YEAR'
    ]
    team_data = team_data.drop(columns=columns_to_drop)
    team_data.to_csv(output_file, index=False)

# Function to preprocess game logs
def preprocess_game_logs(input_file, output_file):
    """Load, clean, and save game log data with specific transformations."""
    game_logs = pd.read_csv(input_file).dropna()

    # Convert 'GAME_DATE' to datetime and sort by date
    game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
    game_logs_sorted = game_logs.sort_values(by='GAME_DATE', ascending=True)

    # Extract 'TEAM1' and 'TEAM2' from 'MATCHUP' column
    game_logs_sorted['TEAM1'] = game_logs_sorted['MATCHUP'].str.split().str[0]
    game_logs_sorted['TEAM2'] = game_logs_sorted['MATCHUP'].str.split().str[2]

    # Drop columns that are not required
    columns_to_drop = ['MATCHUP', 'AVAILABLE_FLAG', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'GAME_ID']
    game_logs_cleaned = game_logs_sorted.drop(columns=columns_to_drop)

    # Convert 'WL' to binary format: 'W' becomes 1, 'L' becomes 0
    game_logs_cleaned['WL'] = game_logs_cleaned['WL'].apply(lambda result: 1 if result == 'W' else 0)

    # Convert 'SEASON_YEAR' to integer format, using only the starting year
    game_logs_cleaned['SEASON_YEAR'] = game_logs_cleaned['SEASON_YEAR'].apply(lambda year: int(year[:4]))

    # Save the processed data to a CSV file
    game_logs_cleaned.to_csv(output_file, index=False)

# Function to preprocess player game logs
def preprocess_player_game_logs(input_file, output_file):
    """Load, clean, and save player game log data with specific transformations."""
    player_game_logs = pd.read_csv(input_file).dropna()

    # Drop unnecessary columns
    columns_to_drop = [
        'PLAYER_NAME', 'NICKNAME', 'TEAM_NAME', 'TEAM_ABBREVIATION', 
        'MATCHUP', 'GAME_DATE', 'GAME_ID'
    ]
    player_game_logs = player_game_logs.drop(columns=columns_to_drop)

    # Save the processed data to a new CSV file
    player_game_logs.to_csv(output_file, index=False)

# File paths for input and output data
team_data_file = 'data/raw/nba_team_data.csv'
team_data_output = 'data/processed/preprocessed_nba_team_data.csv'

game_data_file = 'data/raw/nba_game_logs.csv'
game_data_output = 'data/processed/preprocessed_nba_game_logs.csv'

player_game_logs_file = 'data/raw/nba_player_game_logs.csv'
player_game_logs_output = 'data/processed/preprocessed_nba_player_game_logs.csv'

# Run the preprocessing functions
preprocess_team_data(team_data_file, team_data_output)
print("Team data preprocessed and saved.")

preprocess_game_logs(game_data_file, game_data_output)
print("Game logs preprocessed and saved.")

preprocess_player_game_logs(player_game_logs_file, player_game_logs_output)
print("Player game logs preprocessed and saved.")


