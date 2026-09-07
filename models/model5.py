from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerRegressor(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.feature_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, n_features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)


class TransformerModel:
    name = "Transformer"
    
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, 
                 dropout=0.1, batch_size=64, n_epochs=15, learning_rate=1e-3, 
                 patience=8, model_path=None):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        artifacts_dir = Path(__file__).resolve().parent / "artifacts" / "model5"
        self.model_path = Path(model_path) if model_path else artifacts_dir / "transformer_model.pt"
        
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = None
        self.n_features = None
        
    def fit(self, X, y):
        """
        Train the Transformer model.
        
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
            
        self.n_features = X_features.shape[1]
        
        # Feature scaling
        X_train_arr = self.scaler.fit_transform(X_features.values)
        y_train_arr = y.values.astype(np.float32) if hasattr(y, 'values') else y.astype(np.float32)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_arr, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32).unsqueeze(1)
        
        # Create data loader
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Initialize model
        self.model = TransformerRegressor(
            n_features=self.n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # Training loop with early stopping
        best_val = float('inf')
        wait = 0
        
        # Use a validation split (10% of training data)
        val_size = int(0.1 * len(X_train_tensor))
        X_val_tensor = X_train_tensor[-val_size:]
        y_val_tensor = y_train_tensor[-val_size:]
        X_train_tensor = X_train_tensor[:-val_size]
        y_train_tensor = y_train_tensor[:-val_size]
        
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor), 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        for epoch in range(1, self.n_epochs + 1):
            # Training
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train = float(np.mean(train_losses))
            
            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    val_losses.append(criterion(preds, yb).item())
            avg_val = float(np.mean(val_losses)) if val_losses else float('nan')
            
            # Early stopping
            if avg_val < best_val - 1e-6:
                best_val = avg_val
                wait = 0
                # Save model
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler': self.scaler,
                    'n_features': self.n_features,
                    'feature_cols': self.feature_cols
                }, str(self.model_path))
            else:
                wait += 1
            
            if wait >= self.patience:
                break
        
        # Load best model
        checkpoint = torch.load(str(self.model_path))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: DataFrame with feature columns
        """
        if self.model is None:
            # Try to load from checkpoint
            if self.model_path.exists():
                checkpoint = torch.load(str(self.model_path), map_location=self.device)
                self.n_features = checkpoint.get('n_features')
                self.feature_cols = checkpoint.get('feature_cols')
                self.scaler = checkpoint['scaler']
                self.model = TransformerRegressor(
                    n_features=self.n_features,
                    d_model=self.d_model,
                    nhead=self.nhead,
                    num_layers=self.num_layers,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError("Model not trained and no checkpoint found. Call fit() first.")
        
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
        
        # Feature scaling
        X_test_arr = self.scaler.transform(X_features.values)
        X_test_tensor = torch.tensor(X_test_arr, dtype=torch.float32)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(self.device)
            predictions = self.model(X_test_tensor).cpu().numpy().squeeze()
        
        return predictions

