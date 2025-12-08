import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from .custom_loss import asymmetric_mse_loss


class MLPRegressor(nn.Module):
    def __init__(self, n_features, hidden_layer_sizes=(8, 4), activation="relu"):
        super().__init__()
        layers = []
        input_size = n_features

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


class MLPModel:
    name = "MLP"

    def __init__(
        self,
        hidden_layer_sizes=(8, 4),
        activation="relu",
        batch_size=64,
        n_epochs=500,
        learning_rate=1e-3,
        alpha=2.0,
        random_state=42,
        verbose=True,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.alpha = alpha  # For asymmetric loss
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = None
        self.n_features = None

        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def fit(self, X, y):
        """
        Train the MLP model.

        Args:
            X: DataFrame with feature columns (excluding datetime and target_next_hour)
            y: Target values
        """
        # Select features (exclude datetime and target columns)
        if isinstance(X, pd.DataFrame):
            feature_cols = [
                col for col in X.columns if col not in ["datetime", "target_next_hour"]
            ]
            X_features = X[feature_cols]
            self.feature_cols = feature_cols
        else:
            X_features = X

        self.n_features = X_features.shape[1]

        # Feature scaling
        X_train_arr = self.scaler.fit_transform(X_features.values)
        y_train_arr = (
            y.values.astype(np.float32)
            if hasattr(y, "values")
            else y.astype(np.float32)
        )

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_arr, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self.model = MLPRegressor(
            n_features=self.n_features,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
        ).to(self.device)

        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with progress bar
        self.model.train()

        if self.verbose:
            print(
                f"Training MLP on {self.device} with {len(dataloader)} batches per epoch..."
            )

        epoch_iterator = tqdm(
            range(self.n_epochs),
            desc="Training MLP",
            disable=not self.verbose,
            unit="epoch",
            miniters=1,
            mininterval=0.5,
        )

        for epoch in epoch_iterator:
            epoch_loss = 0.0
            num_batches = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = asymmetric_mse_loss(
                    predictions, batch_y, alpha=self.alpha
                )  # novelty !
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            # Update progress bar (this forces a refresh)
            if self.verbose:
                epoch_iterator.set_postfix({"Loss": f"{avg_loss:.4f}"})
                epoch_iterator.refresh()  # Force immediate update

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
                feature_cols = [
                    col
                    for col in X.columns
                    if col not in ["datetime", "target_next_hour"]
                ]
                X_features = X[feature_cols]
        else:
            X_features = X

        # Feature scaling
        X_test_arr = self.scaler.transform(X_features.values)
        X_test_tensor = torch.tensor(X_test_arr, dtype=torch.float32).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
            predictions = predictions.cpu().numpy()

        return predictions
