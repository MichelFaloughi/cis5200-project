import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class MLPModel:
    name = "MLP"
    
    def __init__(self, hidden_layer_sizes=(200, 100), activation="relu", 
                 solver="adam", max_iter=500, random_state=42):
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=max_iter,
                random_state=random_state,
                verbose=False
            )
        )
        self.feature_cols = None
        
    def fit(self, X, y):
        """
        Train the MLP model.
        
        Args:
            X: DataFrame with feature columns (excluding datetime and target_next_hour)
            y: Target values
        """
        # Select features (exclude datetime and target columns)
        if isinstance(X, pd.DataFrame):
            feature_cols = [col for col in X.columns 
                           if col not in ['datetime', 'target_next_hour']]
            X_features = X[feature_cols]
            self.feature_cols = feature_cols
        else:
            X_features = X
            
        # Train model
        self.model.fit(X_features, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: DataFrame with feature columns
        """
        # Select features
        if isinstance(X, pd.DataFrame):
            if self.feature_cols is not None:
                X_features = X[self.feature_cols]
            else:
                feature_cols = [col for col in X.columns 
                               if col not in ['datetime', 'target_next_hour']]
                X_features = X[feature_cols]
        else:
            X_features = X
        
        # Make predictions
        predictions = self.model.predict(X_features)
        
        return predictions

