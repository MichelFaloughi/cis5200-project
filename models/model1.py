import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXCLUDE_COLUMNS = ['datetime', 'target_next_hour']


class LinearRegressionModel:
    name = "LinearRegression"

    def __init__(self, fit_intercept=True, scaler=True):
        steps = []
        if scaler:
            steps.append(StandardScaler())
        steps.append(LinearRegression(fit_intercept=fit_intercept))

        if len(steps) == 1:
            self.model = steps[0]
        else:
            self.model = make_pipeline(*steps)

        self.feature_cols = None

    def _prepare_features(self, X, fitting=False):
        if isinstance(X, pd.DataFrame):
            if fitting or self.feature_cols is None:
                self.feature_cols = [col for col in X.columns if col not in EXCLUDE_COLUMNS]
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

