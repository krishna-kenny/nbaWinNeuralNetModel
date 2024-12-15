# nbaWinNeuralNetModel

to create the model:
1] run nbaGet.py to get data from api
2] run nbaPreprocessing.py to preprocess the data
3] run nbaModel.py to create final feature data and train the model
4] model is saved in saved_model/model.h5 and saved_model/scaler.pkl

to use the model:
1] run nbaPredict.py
2] enter team name abbreviations
3] returned value is probability of team1 beating team2
