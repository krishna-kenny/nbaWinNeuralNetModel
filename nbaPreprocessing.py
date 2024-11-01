import numpy as np
import pandas as pd
from nbaGet import fetch_team_ids
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations

def collect_team_pairs(seasons):
    """Collect all unique team pairs for each season."""
    team_ids_dict = fetch_team_ids()
    team_pairs = [(team1_id, team2_id, season) for season in seasons for team1_id, team2_id in combinations(team_ids_dict.values(), 2)]
    return team_pairs

def prepare_dataset(team_pairs, team_ids_dict, csv_file):
    """Prepare dataset for training based on team matchup pairs."""
    df = pd.read_csv(csv_file).fillna(0)
    df['TEAM1_ABBR'], df['TEAM2_ABBR'] = df['MATCHUP'].str.split().str[0], df['MATCHUP'].str.split().str[2]
    df['TEAM1_ID'], df['TEAM2_ID'] = df['TEAM1_ABBR'].map(team_ids_dict), df['TEAM2_ABBR'].map(team_ids_dict)

    X, y = [], []
    for team1_id, team2_id, season in team_pairs:
        matchups = df[(df['TEAM1_ID'] == team1_id) & (df['TEAM2_ID'] == team2_id) & (df['SEASON_YEAR'] == season)]
        
        if not matchups.empty:
            features = matchups.iloc[0]
            X.append([
                *[features[f'{stat}_avg_16'] for stat in ['PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG_PCT', 'FTM', 'FTA', 'OREB', 'DREB']],
                features['Win']
            ])
            y.append(features['Win'])
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train a neural network model to predict game outcomes."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    return model
