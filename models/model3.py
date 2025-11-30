import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XGBoostModel:
    name = "XGBoost"
    
    def __init__(self, lag_hours=24, n_estimators=300, learning_rate=0.05, 
                 max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42):
        self.lag_hours = lag_hours
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        self.lag_vars = ["u10", "v10", "t2m", "sp", "wind_speed"]
        self.time_features = ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
        self.feature_cols = None
        
    def _create_lagged_features(self, df):
        """Create lagged features from dataframe."""
        df = df.sort_values("datetime").reset_index(drop=True).copy()
        
        # Construct lagged features
        lag_data = {}
        for var in self.lag_vars:
            for lag in range(self.lag_hours):
                lag_data[f"{var}_lag{lag}"] = df[var].shift(lag)
        
        # Create DataFrame with lagged features
        lag_df = pd.DataFrame(lag_data, index=df.index)
        
        # Combine with time features
        feature_df = pd.concat([lag_df, df[self.time_features]], axis=1)
        
        return feature_df
    
    def fit(self, X, y):
        """
        Train the XGBoost model.
        
        Args:
            X: DataFrame with datetime, features, and target columns
            y: Target values (not used for feature creation, but kept for API consistency)
        """
        # Create lagged features
        X_features = self._create_lagged_features(X)
        
        # Drop rows with NA due to shifting
        valid_mask = ~X_features.isna().any(axis=1)
        X_features = X_features[valid_mask]
        y_clean = y[valid_mask]
        
        # Store feature column names
        self.feature_cols = X_features.columns.tolist()
        
        # Convert to numpy arrays
        X_array = X_features.values
        y_array = y_clean.values
        
        # Train model
        self.model.fit(X_array, y_array, verbose=False)
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: DataFrame with datetime and feature columns
        """
        # Create lagged features
        X_features = self._create_lagged_features(X)
        
        # Drop rows with NA due to shifting
        valid_mask = ~X_features.isna().any(axis=1)
        X_features = X_features[valid_mask]
        
        # Ensure same feature order as training
        if self.feature_cols is not None:
            X_features = X_features[self.feature_cols]
        
        # Convert to numpy array
        X_array = X_features.values
        
        # Make predictions
        predictions = self.model.predict(X_array)
        
        # Create full prediction array (with NaN for invalid rows)
        full_predictions = np.full(len(X), np.nan)
        full_predictions[valid_mask] = predictions
        
        return full_predictions

