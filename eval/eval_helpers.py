import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def asymmetric_mse_loss_numpy(y_pred, y_true, alpha=2.0):
    """
    Compute asymmetric MSE loss for numpy arrays.
    Penalizes under-predictions (y_pred < y_true) more than over-predictions.
    
    Args:
        y_pred: Predicted values (numpy array)
        y_true: True target values (numpy array)
        alpha: Weight factor for under-predictions (alpha > 1)
    
    Returns:
        Asymmetric MSE loss value
    """
    diff = y_pred - y_true
    under_mask = (diff < 0).astype(float)
    weights = 1.0 + (alpha - 1.0) * under_mask
    return np.mean(weights * diff**2)


def compute_metrics(y_true, y_pred, alpha=2.0):
    """
    Compute standard regression metrics and asymmetric MSE loss.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        alpha: Weight factor for asymmetric loss (default 2.0)
    
    Returns:
        dict with 'mae', 'rmse', 'r2', 'asymmetric_mse' keys
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    asym_mse = asymmetric_mse_loss_numpy(y_pred, y_true, alpha=alpha)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'asymmetric_mse': asym_mse
    }


 
# TODO: Here is where we will have to pass our own models
def evaluate(model, X_train, y_train, X_test, y_test, model_name=None, alpha=2.0):
    """
    Evaluate a model by training it and computing metrics.
    
    Args:
        model: Model object with fit() and predict() methods
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name of the model (defaults to model.__class__.__name__)
        alpha: Weight factor for asymmetric loss evaluation (default 2.0)
    
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
    metrics = compute_metrics(y_test_array[valid_mask], y_pred[valid_mask], alpha=alpha)
    
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
    # Use fixed axis limits for consistent comparison across models
    # Wind speeds typically range 0-10 m/s, with some peaks up to 12 m/s
    scatter_min = 0
    scatter_max = 10
    ax2.plot([scatter_min, scatter_max], [scatter_min, scatter_max], 'r--', linewidth=2, label='Perfect')
    ax2.set_xlabel('Actual (m/s)')
    ax2.set_ylabel('Predicted (m/s)')
    ax2.set_title('Scatter Plot')
    ax2.set_xlim([scatter_min, scatter_max])
    ax2.set_ylim([scatter_min, scatter_max])
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


def compare(results_list, plot=True, title_suffix=""):
    """
    Compare multiple models by creating a scorecard table and optional bar chart.
    
    Args:
        results_list: List of result dicts from evaluate()
        plot: Whether to create comparison bar chart
        title_suffix: Optional suffix to add to the title (e.g., " (Calibrated Models)")
    
    Returns:
        pandas DataFrame with comparison metrics
    """
    # Clean model names: remove "(Calibrated)" suffix for cleaner display
    cleaned_results = []
    for result in results_list:
        cleaned_result = result.copy()
        cleaned_name = result['model_name'].replace(' (Calibrated)', '')
        cleaned_result['model_name'] = cleaned_name
        cleaned_results.append(cleaned_result)
    
    # Create scorecard DataFrame
    scorecard_data = []
    for result in cleaned_results:
        metrics = result['metrics']
        scorecard_data.append({
            'Model': result['model_name'],
            'MAE (m/s)': metrics['mae'],
            'RMSE (m/s)': metrics['rmse'],
            'R²': metrics['r2'],
            'Asymmetric MSE': metrics['asymmetric_mse']
        })
    
    scorecard = pd.DataFrame(scorecard_data)
    scorecard = scorecard.sort_values('Asymmetric MSE')  # Sort by asymmetric MSE (lower is better)
    
    # Print table
    print("\n" + "="*60)
    print("MODEL COMPARISON" + title_suffix)
    print("="*60)
    print(scorecard.to_string(index=False))
    print("="*60)
    
    # Optional bar chart
    if plot and len(cleaned_results) > 0:
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        comparison_title = 'Model Comparison' + title_suffix
        fig.suptitle(comparison_title, fontsize=14, fontweight='bold')
        
        model_names = [r['model_name'] for r in cleaned_results]
        maes = [r['metrics']['mae'] for r in results_list]
        rmses = [r['metrics']['rmse'] for r in results_list]
        r2s = [r['metrics']['r2'] for r in results_list]
        asym_mses = [r['metrics']['asymmetric_mse'] for r in results_list]
        
        # MAE
        ax1 = axes[0]
        bars1 = ax1.barh(model_names, maes, color='skyblue', edgecolor='black')
        ax1.set_xlabel('MAE (m/s)')
        ax1.set_title('MAE')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.tick_params(axis='y', labelsize=9)
        for i, (bar, val) in enumerate(zip(bars1, maes)):
            ax1.text(val + max(maes) * 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        # RMSE
        ax2 = axes[1]
        bars2 = ax2.barh(model_names, rmses, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('RMSE (m/s)')
        ax2.set_title('RMSE')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.tick_params(axis='y', labelsize=9)
        for i, (bar, val) in enumerate(zip(bars2, rmses)):
            ax2.text(val + max(rmses) * 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        # R²
        ax3 = axes[2]
        bars3 = ax3.barh(model_names, r2s, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('R²')
        ax3.set_title('R²')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.tick_params(axis='y', labelsize=9)
        for i, (bar, val) in enumerate(zip(bars3, r2s)):
            ax3.text(val + max(r2s) * 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        # Asymmetric MSE
        ax4 = axes[3]
        bars4 = ax4.barh(model_names, asym_mses, color='orange', edgecolor='black')
        ax4.set_xlabel('Asymmetric MSE')
        ax4.set_title('Asymmetric MSE (α=2.0)')
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.tick_params(axis='y', labelsize=9)
        for i, (bar, val) in enumerate(zip(bars4, asym_mses)):
            ax4.text(val + max(asym_mses) * 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    return scorecard

