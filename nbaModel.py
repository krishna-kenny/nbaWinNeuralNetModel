import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


def prepare_dataset(game_logs_file, team_features_file):
    """
    Prepare dataset for training using team-specific features.

    Args:
        game_logs_file: CSV file containing game logs with TEAM1, TEAM2, and WL columns.
        team_features_file: CSV file containing aggregated team features.

    Returns:
        X: Feature matrix for training.
        y: Target vector (win/loss).
    """
    # Load game logs and team features
    game_logs = pd.read_csv(game_logs_file)
    team_features = pd.read_csv(team_features_file)

    # Merge team features for TEAM1 and TEAM2
    game_logs = game_logs.merge(
        team_features,
        how="left",
        left_on="TEAM1",
        right_on="TEAM_ABBREVIATION",
        suffixes=("", "_TEAM1"),
    ).merge(
        team_features,
        how="left",
        left_on="TEAM2",
        right_on="TEAM_ABBREVIATION",
        suffixes=("", "_TEAM2"),
    )

    # Drop unnecessary columns
    game_logs.drop(
        columns=["TEAM_ABBREVIATION", "TEAM_ABBREVIATION_TEAM2"], inplace=True
    )

    # Save features
    game_logs.to_csv("data/features.csv")

    # Handle missing values
    game_logs.fillna(0, inplace=True)

    # Extract features and target
    team1_features = game_logs.filter(regex="_TEAM1$").to_numpy()
    team2_features = game_logs.filter(regex="_TEAM2$").to_numpy()
    X = np.hstack([team1_features, team2_features])
    y = game_logs["WL"].astype(int).to_numpy()

    return X, y


def train_model(X, y):
    """
    Train a neural network model on the given features and labels.

    Args:
        X: Feature matrix for training.
        y: Target vector (win/loss).

    Returns:
        model: Trained neural network model.
    """
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(X.shape[1],)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=50, batch_size=10, validation_split=0.2)

    return model


if __name__ == "__main__":
    game_logs_file = "data/processed/preprocessed_nba_game_logs.csv"
    team_features_file = "data/nba_team_features.csv"

    # Prepare the dataset
    X, y = prepare_dataset(game_logs_file, team_features_file)

    # Check if data is valid for training
    if X.size == 0 or y.size == 0:
        print("No data available to train the model.")
    else:
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
