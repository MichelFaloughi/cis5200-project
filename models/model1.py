import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

EXCLUDE_COLUMNS = ['datetime', 'target_next_hour']


class LinearRegressionModel:
    name = "LinearRegression"

    def __init__(self, degree=2, alpha=10.0, scaler=True):
        steps = []

        if scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
        steps.append(("ridge", Ridge(alpha=alpha)))

        self.model = Pipeline(steps)
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
