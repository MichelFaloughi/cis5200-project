import torch

def asymmetric_mse_loss(y_pred, y_true, alpha=2.0):
    """
    Penalize under-predictions (y_pred < y_true) more than over-predictions.
    alpha > 1 => under-predictions weighted more heavily.
    """
    diff = y_pred - y_true
    under_mask = (diff < 0).float()
    weights = 1.0 + (alpha - 1.0) * under_mask
    return torch.mean(weights * diff**2)