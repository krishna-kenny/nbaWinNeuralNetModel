import numpy as np
import pandas as pd
from nbaGet import fetch_team_ids
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


def collect_team_pairs(seasons):
    """Collect all team pairs for the given seasons."""
    team_ids_dict = fetch_team_ids()
    team_pairs = []

    for season in seasons:
        for team1_abbr, team1_id in team_ids_dict.items():
            for team2_abbr, team2_id in team_ids_dict.items():
                if team1_id != team2_id:
                    team_pairs.append((team1_id, team2_id, season))

    return team_pairs


def prepare_dataset(team_pairs, team_ids_dict, csv_file):
    """Prepare dataset for training based on team matchup pairs."""
    df = pd.read_csv(csv_file)
    df.fillna(0, inplace=True)

    # Add columns for team IDs based on MATCHUP column
    df['TEAM1_ABBR'] = df['MATCHUP'].apply(lambda x: x.split()[0])
    df['TEAM2_ABBR'] = df['MATCHUP'].apply(lambda x: x.split()[2])

    # Map abbreviations to IDs
    df['TEAM1_ID'] = df['TEAM1_ABBR'].map(team_ids_dict)
    df['TEAM2_ID'] = df['TEAM2_ABBR'].map(team_ids_dict)

    X = []
    y = []

    for team1_id, team2_id, season in team_pairs:
        team1_abbr = next((abbr for abbr, id in team_ids_dict.items() if id == team1_id), None)
        team2_abbr = next((abbr for abbr, id in team_ids_dict.items() if id == team2_id), None)

        if team1_abbr is None or team2_abbr is None:
            print(f"Could not find abbreviations for IDs {team1_id} and {team2_id}")
            continue

        matchups = df[(df['TEAM1_ID'] == team1_id) & (df['TEAM2_ID'] == team2_id) & (df['SEASON_YEAR'] == season)]

        if not matchups.empty:
            features = matchups.iloc[0]
            input_data = []

            # 1. Win/Loss History for Past 16 Games (for Both Teams)
            team1_history = features.get('Team1_Win_Loss_History', '[0]*16').strip('[]').split(',')
            team2_history = features.get('Team2_Win_Loss_History', '[0]*16').strip('[]').split(',')
            input_data.extend(map(float, team1_history))
            input_data.extend(map(float, team2_history))

            # 2. Head-to-Head Win/Loss History (Last 8 Games)
            head_to_head_df = df[
                ((df['TEAM1_ID'] == team1_id) & (df['TEAM2_ID'] == team2_id)) |
                ((df['TEAM1_ID'] == team2_id) & (df['TEAM2_ID'] == team1_id))
            ].tail(8)
            head_to_head_history = head_to_head_df['WL'].apply(lambda x: 1 if x == 'W' else 0).tolist()
            head_to_head_history.extend([0] * (8 - len(head_to_head_history)))  # Pad if fewer than 8 games
            input_data.extend(head_to_head_history)

            # 3. Player Points per Minute (Last 16 Games for Each Team)
            team1_ppm = features.get('Team1_PPM', '[0]*15').strip('[]').split(',')
            team2_ppm = features.get('Team2_PPM', '[0]*15').strip('[]').split(',')
            input_data.extend(map(float, team1_ppm))
            input_data.extend(map(float, team2_ppm))

            # 4. Average Team Stats (Last 16 Games)
            team1_stats = features.get('Team1_Avg_Stats', '[0]*10').strip('[]').split(',')
            team2_stats = features.get('Team2_Avg_Stats', '[0]*10').strip('[]').split(',')
            input_data.extend(map(float, team1_stats))
            input_data.extend(map(float, team2_stats))

            # 5. Opponent Defensive Strength (Last 16 Games)
            team1_opp_def = features.get('Team1_Opp_Defense', '[0]*5').strip('[]').split(',')
            team2_opp_def = features.get('Team2_Opp_Defense', '[0]*5').strip('[]').split(',')
            input_data.extend(map(float, team1_opp_def))
            input_data.extend(map(float, team2_opp_def))

            X.append(input_data)
            y.append(1 if features['WL'] == 'W' else 0)

    print(f"Collected {len(X)} samples for training.")
    return np.array(X), np.array(y)


def train_model(X, y):
    """Train a neural network model on the given features and labels."""
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=10, validation_split=0.2)

    return model


if __name__ == "__main__":
    seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
    team_ids_dict = fetch_team_ids()
    
    team_pairs = collect_team_pairs(seasons)

    csv_file = 'data/processed_nba_game_logs.csv'
    X, y = prepare_dataset(team_pairs, team_ids_dict, csv_file)

    if X.size == 0 or y.size == 0:
        print("No data available to train the model.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = train_model(X_train, y_train)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
