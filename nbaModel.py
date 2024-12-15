import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def prepare_dataset(game_logs_file, features_file):
    """
    Prepare dataset for training using team-specific features.

    Args:
        game_logs_file: CSV file containing game logs with TEAM1, TEAM2, and WL columns.
        features_file: CSV file to save aggregated team features.

    Returns:
        X: Feature matrix for training.
        y: Target vector (win/loss).
    """
    # Load game logs
    game_logs = pd.read_csv(game_logs_file)

    # Aggregate features by team
    team_features = (
        game_logs.groupby("TEAM1")
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"TEAM1": "TEAM"})
    )

    # Save the aggregated features to features_file
    team_features.to_csv(features_file, index=False)

    # Merge team features for TEAM1 and TEAM2
    game_logs = game_logs.merge(
        team_features,
        how="left",
        left_on="TEAM1",
        right_on="TEAM",
        suffixes=("", "_TEAM1"),
    ).merge(
        team_features,
        how="left",
        left_on="TEAM2",
        right_on="TEAM",
        suffixes=("", "_TEAM2"),
    )

    # Drop unnecessary columns
    columns_to_drop = [
        "TEAM",
        "TEAM_TEAM2",
        "TEAM_CONFERENCE",
        "TEAM_SLUG",
        "TEAM_CONFERENCE_TEAM2",
        "TEAM_SLUG_TEAM2",
        "PLAYER_ID",
        "PLAYER_ID_TEAM2",
        "AVAILABLE_FLAG",
        "AVAILABLE_FLAG_TEAM2",
        "GAME_TIMESTAMP",
        "GAME_TIMESTAMP_TEAM2",
    ]
    game_logs.drop(
        columns=[col for col in columns_to_drop if col in game_logs.columns],
        inplace=True,
    )

    # Feature engineering: Create new features for differences and ratios
    numeric_columns = game_logs.filter(regex="_TEAM1$").columns
    diff_features = {}
    ratio_features = {}
    for col in numeric_columns:
        base_col = col.replace("_TEAM1", "")
        diff_features[f"{base_col}_DIFF"] = (
            game_logs[f"{base_col}_TEAM1"] - game_logs[f"{base_col}_TEAM2"]
        )
        ratio_features[f"{base_col}_RATIO"] = game_logs[f"{base_col}_TEAM1"] / (
            game_logs[f"{base_col}_TEAM2"] + 1e-5
        )

    # Add all new features at once to optimize performance
    new_features = pd.concat(
        [pd.DataFrame(diff_features), pd.DataFrame(ratio_features)], axis=1
    )
    game_logs = pd.concat([game_logs, new_features], axis=1)

    # Handle missing values
    game_logs.fillna(0, inplace=True)

    # Extract features and target
    feature_columns = game_logs.select_dtypes(include=np.number).columns.difference(
        ["WL"]
    )
    X = game_logs[feature_columns].to_numpy()
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
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Define the neural network
    model = Sequential()
    model.add(Dense(256, activation="relu", input_shape=(X_resampled.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X_resampled, y_resampled, epochs=16, batch_size=32, validation_split=0.2)

    return model, scaler


if __name__ == "__main__":
    game_logs_file = "data/processed/preprocessed_nba_game_logs.csv"
    features_file = "data/features.csv"

    # Prepare the dataset
    X, y = prepare_dataset(game_logs_file, features_file)

    # Check if data is valid for training
    if X.size == 0 or y.size == 0:
        print("No data available to train the model.")
    else:
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the neural network
        model, scaler = train_model(X_train, y_train)

        # Evaluate the neural network
        X_test_scaled = scaler.transform(X_test)
        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # Train and evaluate a Random Forest for comparison
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest Accuracy: {rf_accuracy}")
