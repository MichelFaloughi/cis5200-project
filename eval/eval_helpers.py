import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    """
    Compute standard regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        dict with 'mae', 'rmse', 'r2' keys
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


 
# TODO: Here is where we will have to pass our own models
def evaluate(model, X_train, y_train, X_test, y_test, model_name=None):
    """
    Evaluate a model by training it and computing metrics.
    
    Args:
        model: Model object with fit() and predict() methods
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name of the model (defaults to model.__class__.__name__)
    
    Returns:
        dict with 'model_name', 'metrics', 'predictions' keys
    """
    # Get model name
    if model_name is None:
        model_name = getattr(model, 'name', model.__class__.__name__)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Ensure predictions are numpy array
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    y_pred = np.array(y_pred).flatten()
    
    # Handle NaN predictions (e.g., from lagged features)
    # Filter out NaN values for metric computation
    valid_mask = ~np.isnan(y_pred)
    if isinstance(y_test, pd.Series):
        y_test_array = y_test.values
    else:
        y_test_array = np.array(y_test).flatten()
    
    if valid_mask.sum() == 0:
        raise ValueError(f"All predictions are NaN for {model_name}")
    
    # Compute metrics only on valid predictions
    metrics = compute_metrics(y_test_array[valid_mask], y_pred[valid_mask])
    
    # Update predictions to keep NaN for alignment with original test set
    # (This allows plotting to handle NaN values appropriately)
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'predictions': y_pred
    }


def plot_predictions(y_true, y_pred, model_name="Model", datetime_index=None, figsize=(12, 4)):
    """
    Create visualization plots for predictions.
    
    Creates 3 subplots:
    1. Time series (predicted vs actual)
    2. Scatter plot (predicted vs actual)
    3. Residual plot
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model (for title)
        datetime_index: Optional datetime index for time series plot
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'{model_name}', fontsize=14, fontweight='bold')
    
    # Sample if too many points for time series
    if len(y_true) > 1000:
        plot_indices = np.arange(1000)
        y_true_plot = y_true.iloc[plot_indices] if hasattr(y_true, 'iloc') else y_true[plot_indices]
        y_pred_plot = y_pred[plot_indices]
        if datetime_index is not None:
            datetime_plot = datetime_index.iloc[plot_indices] if hasattr(datetime_index, 'iloc') else datetime_index[plot_indices]
        else:
            datetime_plot = None
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        datetime_plot = datetime_index
    
    residuals = y_true_plot - y_pred_plot
    
    # 1. Time series
    ax1 = axes[0]
    if datetime_plot is not None:
        ax1.plot(datetime_plot, y_true_plot, label='Actual', alpha=0.8, linewidth=2, color='#2E86AB')
        ax1.plot(datetime_plot, y_pred_plot, label='Predicted', alpha=0.8, linewidth=2, color='#A23B72', linestyle='--')
        ax1.set_xlabel('Time')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax1.plot(y_true_plot.values if hasattr(y_true_plot, 'values') else y_true_plot, 
                 label='Actual', alpha=0.8, linewidth=2, color='#2E86AB')
        ax1.plot(y_pred_plot, label='Predicted', alpha=0.8, linewidth=2, color='#A23B72', linestyle='--')
        ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Wind Speed (m/s)')
    ax1.set_title('Time Series')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true_plot, y_pred_plot, alpha=0.5, s=20)
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax2.set_xlabel('Actual (m/s)')
    ax2.set_ylabel('Predicted (m/s)')
    ax2.set_title('Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual plot
    ax3 = axes[2]
    ax3.scatter(y_pred_plot, residuals, alpha=0.5, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted (m/s)')
    ax3.set_ylabel('Residuals (m/s)')
    ax3.set_title('Residuals')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare(results_list, plot=True):
    """
    Compare multiple models by creating a scorecard table and optional bar chart.
    
    Args:
        results_list: List of result dicts from evaluate()
        plot: Whether to create comparison bar chart
    
    Returns:
        pandas DataFrame with comparison metrics
    """
    # Create scorecard DataFrame
    scorecard_data = []
    for result in results_list:
        metrics = result['metrics']
        scorecard_data.append({
            'Model': result['model_name'],
            'MAE (m/s)': metrics['mae'],
            'RMSE (m/s)': metrics['rmse'],
            'R²': metrics['r2']
        })
    
    scorecard = pd.DataFrame(scorecard_data)
    scorecard = scorecard.sort_values('RMSE (m/s)')  # Sort by RMSE (lower is better)
    
    # Print table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(scorecard.to_string(index=False))
    print("="*60)
    
    # Optional bar chart
    if plot and len(results_list) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')
        
        model_names = [r['model_name'] for r in results_list]
        maes = [r['metrics']['mae'] for r in results_list]
        rmses = [r['metrics']['rmse'] for r in results_list]
        r2s = [r['metrics']['r2'] for r in results_list]
        
        # MAE
        ax1 = axes[0]
        bars1 = ax1.barh(model_names, maes, color='skyblue', edgecolor='black')
        ax1.set_xlabel('MAE (m/s)')
        ax1.set_title('MAE')
        ax1.grid(True, alpha=0.3, axis='x')
        for i, (bar, val) in enumerate(zip(bars1, maes)):
            ax1.text(val + max(maes) * 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        # RMSE
        ax2 = axes[1]
        bars2 = ax2.barh(model_names, rmses, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('RMSE (m/s)')
        ax2.set_title('RMSE')
        ax2.grid(True, alpha=0.3, axis='x')
        for i, (bar, val) in enumerate(zip(bars2, rmses)):
            ax2.text(val + max(rmses) * 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        # R²
        ax3 = axes[2]
        bars3 = ax3.barh(model_names, r2s, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('R²')
        ax3.set_title('R²')
        ax3.grid(True, alpha=0.3, axis='x')
        for i, (bar, val) in enumerate(zip(bars3, r2s)):
            ax3.text(val + max(r2s) * 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    return scorecard

