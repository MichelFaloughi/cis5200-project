import numpy as np

class PersistenceModel:
    name = "PersistenceBaseline"

    def fit(self, X, y):
        """
        Nothing to learn — baseline model.
        """
        return self
    
    def predict(self, X):
        """
        Predict next-hour wind speed = current hour wind speed.
        X must include a 'wind_speed' column.
        """
        return X["wind_speed"].values

