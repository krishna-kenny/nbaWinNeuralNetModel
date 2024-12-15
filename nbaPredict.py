import numpy as np
import pandas as pd
from nba_api.stats.static import teams
from nbaModel import (
    load_trained_model,
)  # Ensure this is correctly implemented in your nbaModel.py


# Get Team ID based on abbreviation
def get_team_id_by_abbreviation(team_abbreviation):
    """Retrieve the team ID by abbreviation."""
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if team["abbreviation"].lower() == team_abbreviation.lower():
            return team["id"]
    raise ValueError(
        f"Team '{team_abbreviation}' not found! Please enter a valid abbreviation."
    )


# Fetch team-specific features
def fetch_team_features(team_abbreviation, features_file):
    """
    Retrieve the team-specific features for the given team abbreviation.
    Args:
        team_abbreviation: Abbreviation of the NBA team (e.g., 'LAL').
        features_file: CSV file containing the aggregated team features.
    Returns:
        numpy array of the team's features.
    """
    team_features = pd.read_csv(features_file)
    team_row = team_features[team_features["TEAM"] == team_abbreviation.upper()]
    if team_row.empty:
        raise ValueError(
            f"Features for team '{team_abbreviation}' not found in {features_file}."
        )
    return team_row.drop(columns=["TEAM"]).to_numpy().flatten()


# Predict win probability
def predict_matchup_win_probability(
    team1_abbreviation, team2_abbreviation, features_file
):
    """
    Predict the win probability of Team 1 beating Team 2.
    Args:
        team1_abbreviation: Abbreviation of Team 1 (e.g., 'LAL').
        team2_abbreviation: Abbreviation of Team 2 (e.g., 'BOS').
        features_file: Path to the CSV file containing aggregated team features.
    """
    # Load the trained model and scaler
    model, scaler = load_trained_model()

    # Fetch features for both teams
    team1_features = fetch_team_features(team1_abbreviation, features_file)
    team2_features = fetch_team_features(team2_abbreviation, features_file)

    # Calculate difference and ratio features
    diff_features = team1_features - team2_features
    ratio_features = team1_features / (team2_features + 1e-5)

    # Combine features for the model
    matchup_features = np.concatenate([diff_features, ratio_features]).reshape(1, -1)

    # Debug: Check input shape
    print(f"Matchup features shape: {matchup_features.shape}")
    print(f"Scaler expects: {scaler.n_features_in_}")

    # Ensure consistent feature count
    if matchup_features.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Feature count mismatch. Got {matchup_features.shape[1]} features, "
            f"but scaler expects {scaler.n_features_in_}. Check feature engineering consistency."
        )

    # Scale the features
    scaled_features = scaler.transform(matchup_features)

    # Predict win probability for Team 1
    win_probability = model.predict(scaled_features)[0][0]
    print(
        f"Predicted probability of {team1_abbreviation} beating {team2_abbreviation}: {win_probability * 100:.2f}%"
    )
    return win_probability


def display_team_data():
    """Display team names, abbreviations, and IDs in a 2D array."""
    nba_teams = teams.get_teams()
    team_data = np.array(
        [[team["full_name"], team["abbreviation"], team["id"]] for team in nba_teams]
    )
    print("\nAvailable Teams:")
    print(pd.DataFrame(team_data, columns=["Team Name", "Abbreviation", "Team ID"]))


def main():
    """Main function to handle user input and prediction."""
    features_file = "data/features.csv"  # Path to the features file

    # Display team data before taking user input
    display_team_data()

    team1_abbreviation = input(
        "Enter Team 1 abbreviation (e.g., 'LAL' for Los Angeles Lakers): "
    ).strip()
    team2_abbreviation = input(
        "Enter Team 2 abbreviation (e.g., 'BOS' for Boston Celtics): "
    ).strip()

    try:
        predict_matchup_win_probability(
            team1_abbreviation, team2_abbreviation, features_file
        )
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
