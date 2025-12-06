import numpy as np
import torch

from sklearn.base import BaseEstimator, RegressorMixin


# ---------------------------------------------------------------------
# 1. Custom asymmetric loss (PyTorch)
# ---------------------------------------------------------------------

# TODO: we already have this in custom_loss.py
def asymmetric_mse_loss(y_pred: torch.Tensor,
                        y_true: torch.Tensor,
                        alpha: float = 2.0) -> torch.Tensor:
    """
    Penalize under-predictions (y_pred < y_true) more than over-predictions.
    alpha > 1 => under-predictions weighted more heavily.

    Args:
        y_pred: model predictions (tensor)
        y_true: ground truth values (tensor)
        alpha:  weight multiplier for under-predictions

    Returns:
        Scalar tensor: asymmetric MSE loss.
    """
    diff = y_pred - y_true                  # error
    under_mask = (diff < 0).float()         # 1 where underpredict
    weights = 1.0 + (alpha - 1.0) * under_mask
    return torch.mean(weights * diff ** 2)


# ---------------------------------------------------------------------
# 2. Post-hoc wrapper: learns y' = a * y_base + b per model
# ---------------------------------------------------------------------

class AsymmetricPostHocRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps a pre-trained regression model and learns an affine transform:

        y_adj = a * y_pred_base + b

    to minimize asymmetric_mse_loss on a given dataset (X, y).

    This does NOT retrain the base model; it only tunes (a, b).
    """

    def __init__(
        self,
        base_model,
        alpha: float = 2.0,
        lr: float = 1e-2,
        n_steps: int = 500,
        verbose: bool = False,
        min_improvement: float = 0.0
    ):
        """
        Args:
            base_model:  Fitted sklearn-like regressor with .predict().
            alpha:       Asymmetry parameter for the loss.
            lr:          Learning rate for optimizing a, b.
            n_steps:     Number of gradient steps.
            verbose:     If True, prints optimization progress.
            min_improvement: Only keep transform if it improves asymmetric
                            loss by at least this amount; otherwise use base.
        """
        self.base_model = base_model
        self.alpha = alpha
        self.lr = lr
        self.n_steps = n_steps
        self.verbose = verbose
        self.min_improvement = min_improvement

        self.a_ = None
        self.b_ = None
        self.use_transform_ = True  # set in fit()

    def fit(self, X, y):
        """
        Fit the post-hoc transform using validation data X, y.

        Assumes base_model is already trained.
        We:
        1) Get base_model predictions on X
        2) Drop any samples where predictions or targets are NaN/inf
        3) Optimize a, b in y_adj = a * y_pred + b to minimize asymmetric_mse_loss
        4) Keep the transform only if it improves asymmetric loss by at least
            self.min_improvement; otherwise fall back to identity.
        """
        # 1) Base predictions
        y_pred_np = self.base_model.predict(X)
        y_true_np = np.asarray(y, dtype=float)

        # 2) Filter out NaNs / infs
        mask = np.isfinite(y_pred_np) & np.isfinite(y_true_np)
        num_valid = mask.sum()

        if num_valid == 0:
            if self.verbose:
                print(
                    "[AsymmetricPostHocRegressor] All predictions or targets are "
                    "NaN/inf; skipping transform and using identity."
                )
            self.a_ = 1.0
            self.b_ = 0.0
            self.use_transform_ = False
            return self

        if num_valid < len(y_pred_np) and self.verbose:
            print(
                f"[AsymmetricPostHocRegressor] Warning: dropping "
                f"{len(y_pred_np) - num_valid} samples with NaN/inf for post-hoc fit."
            )

        y_pred_np = y_pred_np[mask]
        y_true_np = y_true_np[mask]

        # 3) Convert to tensors
        y_pred = torch.tensor(y_pred_np, dtype=torch.float32)
        y_true = torch.tensor(y_true_np, dtype=torch.float32)

        # 4) Baseline asymmetric loss (no transform)
        with torch.no_grad():
            base_loss = asymmetric_mse_loss(y_pred, y_true, alpha=self.alpha).item()

        # 5) Parameters for affine transform: y_adj = a * y_pred + b
        a = torch.nn.Parameter(torch.tensor(1.0))
        b = torch.nn.Parameter(torch.tensor(0.0))

        optimizer = torch.optim.Adam([a, b], lr=self.lr)

        for step in range(self.n_steps):
            optimizer.zero_grad()
            y_adj = a * y_pred + b
            loss = asymmetric_mse_loss(y_adj, y_true, alpha=self.alpha)
            loss.backward()
            optimizer.step()

            if self.verbose and step % 100 == 0:
                print(
                    f"[{self.__class__.__name__}] step={step:3d} "
                    f"loss={loss.item():.6f} a={a.item():.4f} b={b.item():.4f}"
                )

        # 6) Final transformed loss
        with torch.no_grad():
            final_loss = asymmetric_mse_loss(a * y_pred + b, y_true, alpha=self.alpha).item()

        if self.verbose:
            print(
                f"[{self.__class__.__name__}] base_loss={base_loss:.6f}, "
                f"posthoc_loss={final_loss:.6f}"
            )

        # 7) Decide whether to use transform
        if final_loss < base_loss - self.min_improvement:
            # Transformation helped
            self.a_ = float(a.detach().item())
            self.b_ = float(b.detach().item())
            self.use_transform_ = True
            if self.verbose:
                print(f"Using post-hoc transform: a={self.a_:.4f}, b={self.b_:.4f}")
        else:
            # Transformation didn't help enough; fall back to identity
            self.a_ = 1.0
            self.b_ = 0.0
            self.use_transform_ = False
            if self.verbose:
                print(
                    "Post-hoc transform did not improve loss enough; "
                    "using base model outputs (identity transform)."
                )

        return self


    def predict(self, X):
        """
        Apply base_model.predict, then the learned affine transform if enabled.
        """
        base_pred = self.base_model.predict(X)

        if self.a_ is None or self.b_ is None:
            raise RuntimeError("You must call fit() before predict().")

        return self.a_ * base_pred + self.b_


# ---------------------------------------------------------------------
# 3. Utility: evaluate asymmetric loss for any sklearn regressor
# ---------------------------------------------------------------------

def eval_asymmetric_loss(model, X, y, alpha: float = 2.0) -> float:
    """
    Compute asymmetric_mse_loss for a sklearn-like regressor,
    ignoring NaNs / infs in predictions or targets.
    """
    y_pred = model.predict(X)
    y_true = np.asarray(y, dtype=float)

    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() == 0:
        return float("nan")

    y_pred_t = torch.tensor(y_pred[mask], dtype=torch.float32)
    y_true_t = torch.tensor(y_true[mask], dtype=torch.float32)
    return float(asymmetric_mse_loss(y_pred_t, y_true_t, alpha=alpha).item())

