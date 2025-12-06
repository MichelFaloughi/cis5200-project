import pandas as pd
from sklearn.ensemble import RandomForestRegressor

EXCLUDE_COLUMNS = ['datetime', 'target_next_hour']


class RandomForestModel:
    name = "RandomForest"

    def __init__(self,
                 n_estimators=400,
                 max_depth=20,
                 max_features=None,
                 min_samples_split=2,
                 min_samples_leaf=2,
                 random_state=42,
                 n_jobs=-1):

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_cols = None

    def _prepare_features(self, X, fitting=False):
        if isinstance(X, pd.DataFrame):
            if fitting or self.feature_cols is None:
                self.feature_cols = [
                    col for col in X.columns if col not in EXCLUDE_COLUMNS
                ]
            X_features = X[self.feature_cols]
        else:
            X_features = X
        return X_features

    def fit(self, X, y):
        X_features = self._prepare_features(X, fitting=True)
        self.model.fit(X_features, y)
        return self

    def predict(self, X):
        X_features = self._prepare_features(X)
        return self.model.predict(X_features)
