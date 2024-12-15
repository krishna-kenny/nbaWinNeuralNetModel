import pandas as pd
import numpy as np


# Function to preprocess team data
def preprocess_team_data(input_file, output_file):
    """Load, clean, and save team data by removing unnecessary columns."""
    team_data = pd.read_csv(input_file).dropna()
    columns_to_drop = [
        "TEAM_NAME",
        "TEAM_CITY",
        "SEASON_YEAR",
        "TEAM_CODE",
        "TEAM_DIVISION",
        "MIN_YEAR",
        "MAX_YEAR",
    ]
    team_data = team_data.drop(columns=columns_to_drop)
    team_data = team_data.drop_duplicates()
    team_data.to_csv(output_file, index=False)
    print(f"Team data preprocessed and saved. rows: {team_data.shape}")


# Function to preprocess game logs
def preprocess_game_logs(input_file, output_file):
    """Load, clean, and save game log data with specific transformations."""
    game_logs = pd.read_csv(input_file).dropna()

    # Convert 'GAME_DATE' to datetime and sort by date
    game_logs["GAME_DATE"] = pd.to_datetime(game_logs["GAME_DATE"])
    game_logs_sorted = game_logs.sort_values(by="GAME_DATE", ascending=True)

    # Extract 'TEAM1' and 'TEAM2' from 'MATCHUP' column
    game_logs_sorted["TEAM1"] = game_logs_sorted["MATCHUP"].str.split().str[0]
    game_logs_sorted["TEAM2"] = game_logs_sorted["MATCHUP"].str.split().str[2]

    # Drop columns that are not required
    columns_to_drop = [
        "MATCHUP",
        "AVAILABLE_FLAG",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "GAME_ID",
    ]
    game_logs_cleaned = game_logs_sorted.drop(columns=columns_to_drop)

    # Convert 'WL' to binary format: 'W' becomes 1, 'L' becomes 0
    game_logs_cleaned["WL"] = game_logs_cleaned["WL"].apply(
        lambda result: 1 if result == "W" else 0
    )

    # Convert 'SEASON_YEAR' to integer format, using only the starting year
    game_logs_cleaned["SEASON_YEAR"] = game_logs_cleaned["SEASON_YEAR"].apply(
        lambda year: int(year[:4])
    )

    game_logs_cleaned.to_csv(output_file, index=False)
    print(f"Game logs preprocessed and saved. rows: {game_logs_cleaned.shape}")


# Function to preprocess player game logs
def preprocess_player_game_logs(input_file, output_file):
    """Load, clean, and save player game log data with specific transformations."""
    player_game_logs = pd.read_csv(input_file).dropna()

    # Drop unnecessary columns
    columns_to_drop = [
        "PLAYER_NAME",
        "NICKNAME",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "GAME_ID",
        "MIN_SEC",
    ]
    player_game_logs = player_game_logs.drop(columns=columns_to_drop)
    player_game_logs["WL"] = player_game_logs["WL"].apply(
        lambda x: 1 if x == "W" else 0
    )
    player_game_logs["SEASON_YEAR"] = player_game_logs["SEASON_YEAR"].apply(
        lambda x: x[:4]
    )

    player_game_logs.to_csv(output_file, index=False)
    print(f"Player game logs preprocessed and saved. rows: {player_game_logs.shape}")


# Function to compute weighted averages
def compute_weighted_avg(player_id, df):
    """Compute weighted averages for a given player."""
    player_rows = df[df["PLAYER_ID"] == player_id].copy()

    # Retain TEAM_ID
    team_id = player_rows["TEAM_ID"].iloc[0]

    # Convert GAME_DATE to a timestamp for weighting
    player_rows["GAME_TIMESTAMP"] = pd.to_datetime(player_rows["GAME_DATE"]).apply(
        lambda x: x.timestamp()
    )
    max_timestamp = player_rows["GAME_TIMESTAMP"].max()

    # Calculate weights based on recency
    player_rows["WEIGHTS"] = np.exp(
        (player_rows["GAME_TIMESTAMP"] - max_timestamp) / 1e7
    )
    player_rows["WEIGHTS"] /= player_rows["WEIGHTS"].sum()

    # Compute weighted average for all columns after WL
    weighted_avg = (
        player_rows.iloc[:, df.columns.get_loc("WL") + 1 :]  # Select columns after WL
        .mul(player_rows["WEIGHTS"], axis=0)  # Multiply each column by weights
        .sum()  # Sum the weighted values for each column
    )
    weighted_avg["TEAM_ID"] = team_id  # Include TEAM_ID in the output
    return weighted_avg


# Function to create feature data
def create_feature_data(input_csv, output_csv):
    """Generate feature data where each player is represented by a single row."""
    # Load the dataset
    df = pd.read_csv(input_csv)
    df.fillna(0, inplace=True)

    # Ensure GAME_DATE is parsed correctly
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Get unique players
    unique_players = df["PLAYER_ID"].unique()

    # Create a new dataframe to store the features
    feature_data = []

    for player_id in unique_players:
        weighted_avg = compute_weighted_avg(player_id, df)
        weighted_avg["PLAYER_ID"] = player_id  # Retain the player ID
        feature_data.append(weighted_avg)

    # Convert the list to a DataFrame
    feature_df = pd.DataFrame(feature_data)

    # Save to CSV
    feature_df.to_csv(output_csv, index=False)
    print(f"Feature data saved to {output_csv}. rows: {feature_df.shape}")


# Function to compute team-level aggregated features
def create_team_features(player_features_file, team_features_output):
    """
    Generate aggregated features for each team by averaging player statistics.
    """
    import pandas as pd

    # Load the player features data
    player_data = pd.read_csv(player_features_file)

    # Compute team features
    team_features = player_data.groupby("TEAM_ID").mean().reset_index()

    # Load mapping of TEAM_ID to TEAM_ABBREVIATION and additional features
    preprocessed_nba_team_data = pd.read_csv(
        "data/processed/preprocessed_nba_team_data.csv"
    )
    id_to_abbr_map = preprocessed_nba_team_data.set_index("TEAM_ID")[
        "TEAM_ABBREVIATION"
    ].to_dict()

    # Apply a function to map TEAM_ID to ABBR
    team_features["TEAM_ID"] = team_features["TEAM_ID"].map(id_to_abbr_map)

    # Rename the column
    team_features.rename(columns={"TEAM_ID": "TEAM_ABBREVIATION"}, inplace=True)

    # Merge additional team features
    additional_features = preprocessed_nba_team_data.drop(
        columns=["TEAM_ID", "TEAM_CONFERENCE", "TEAM_SLUG"]
    )
    team_features = team_features.merge(
        additional_features,
        left_on="TEAM_ABBREVIATION",
        right_on="TEAM_ABBREVIATION",
        how="left",
    )

    # Save the resulting team features to a CSV file
    team_features.to_csv(team_features_output, index=False)
    print(
        f"Team feature data saved to {team_features_output}. rows: {team_features.shape}"
    )


# File paths for input and output data
team_data_file = "data/raw/nba_team_data.csv"
team_data_output = "data/processed/preprocessed_nba_team_data.csv"

game_data_file = "data/raw/nba_game_logs.csv"
game_data_output = "data/processed/preprocessed_nba_game_logs.csv"

player_game_logs_file = "data/raw/nba_player_game_logs.csv"
player_game_logs_output = "data/processed/preprocessed_nba_player_game_logs.csv"

# Run the preprocessing functions
preprocess_team_data(team_data_file, team_data_output)
preprocess_game_logs(game_data_file, game_data_output)
preprocess_player_game_logs(player_game_logs_file, player_game_logs_output)
create_feature_data(player_game_logs_output, "data/nba_player_features.csv")

# File paths for player features and team output
player_features_file = "data/nba_player_features.csv"
team_features_output = "data/nba_team_features.csv"

# Run the team feature creation function
create_team_features(player_features_file, team_features_output)
