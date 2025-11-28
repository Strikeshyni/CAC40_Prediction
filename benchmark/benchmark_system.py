import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_lstm_v1 import model_v1
from models.model_lstm_v2 import model_lstm_v2
from models.model_xgboost import model_xgboost
from models.model_transformer import model_transformer

def evaluate_horizon(model_wrapper, horizon=30, num_samples=50):
    """
    Evaluates the model's recursive prediction performance over a horizon.
    
    Args:
        model_wrapper: Instance of model wrapper class
        horizon: Number of days to predict into the future (recursive)
        num_samples: Number of starting points to test (randomly selected from test set)
        
    Returns:
        dict: {step (1..horizon): mse_value}
    """
    scaler = model_wrapper.scaler
    X_test = model_wrapper.X_test
    y_test = model_wrapper.y_test
    
    # Ensure we have enough data
    max_start_idx = len(y_test) - horizon
    if max_start_idx <= 0:
        print("Not enough test data for this horizon.")
        return {}
    
    # Select random starting indices
    # We use a fixed seed for reproducibility if needed, or just random
    np.random.seed(42)
    start_indices = np.linspace(0, max_start_idx, num=min(num_samples, max_start_idx), dtype=int)
    
    mse_per_step = {k: [] for k in range(1, horizon + 1)}
    
    print(f"Evaluating on {len(start_indices)} samples...")
    
    for start_idx in start_indices:
        # Initial sequence
        current_seq = X_test[start_idx].copy() # Shape (time_step,)
        
        for step in range(1, horizon + 1):
            # Predict
            if hasattr(model_wrapper, 'predict'):
                # New models have a predict method that handles reshaping and inverse transform
                # Expects (samples, time_step)
                input_seq = current_seq.reshape(1, -1)
                pred_val = model_wrapper.predict(input_seq)[0][0]
                
                # For recursive step, we need the scaled value to append to sequence
                # We need to re-scale the prediction to feed it back
                pred_val_scaled = scaler.transform([[pred_val]])[0][0]
            else:
                # Legacy model_v1
                # Prepare input (1, time_step, 1)
                input_seq = current_seq.reshape(1, len(current_seq), 1)
                pred_scaled = model_wrapper.best_model.predict(input_seq, verbose=0)
                pred_val_scaled = pred_scaled[0][0]
                pred_val = scaler.inverse_transform([[pred_val_scaled]])[0][0]
            
            # Get Actual
            # y_test[start_idx] is target for step 1
            # y_test[start_idx + step - 1] is target for step 'step'
            actual_val_scaled = y_test[start_idx + step - 1]
            actual_val = scaler.inverse_transform([[actual_val_scaled]])[0][0]
            
            sq_error = (pred_val - actual_val) ** 2
            mse_per_step[step].append(sq_error)
            
            # Update sequence for next step (Recursive)
            # Remove first element, append predicted value (scaled)
            current_seq = np.append(current_seq[1:], pred_val_scaled)
            
    # Average MSE
    avg_mse_per_step = {k: np.mean(v) for k, v in mse_per_step.items()}
    return avg_mse_per_step

def run_benchmark():
    stocks = ['ENGI.PA', 'AIR.PA', 'SAN.PA', 'BNP.PA'] # Example CAC 40 stocks
    models_to_test = [model_lstm_v2] #, model_v1, model_lstm_v2, model_xgboost, model_transformer]
    horizon = 30
    
    results = {}
    
    # Define date range for training/testing
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d') # 5 years history
    
    for stock in stocks:
        for ModelClass in models_to_test:
            model_name_str = ModelClass.__name__
            run_id = f"{stock}_{model_name_str}"
            print(f"\n=== Benchmarking {run_id} ===")
            
            try:
                print(f"Training {model_name_str} for {stock}...")
                
                # Prepare kwargs based on model type
                kwargs = {
                    'model_name': f"benchmark_{run_id}",
                    'stock_name': stock,
                    'from_date': start_date,
                    'to_date': end_date,
                    'train_size_percent': 0.7,
                    'val_size_percent': 0.15, # Leaves 0.15 for test
                    'time_step_train_split': 60,
                    'global_tuning': False
                }
                
                if model_name_str == 'model_xgboost':
                    kwargs['n_iter'] = 2
                else:
                    kwargs['epochs'] = 5
                
                model = ModelClass(**kwargs)
                
                # Evaluate
                print("Evaluating horizon...")
                mse_curve = evaluate_horizon(model, horizon=horizon, num_samples=30)
                results[run_id] = mse_curve
                
                # Print summary
                print(f"MSE @ 1 days: {mse_curve.get(1, 'N/A'):.2f}")
                print(f"MSE @ 5 days: {mse_curve.get(5, 'N/A'):.2f}")
                print(f"MSE @ 10 days: {mse_curve.get(10, 'N/A'):.2f}")
                print(f"MSE @ 30 days: {mse_curve.get(30, 'N/A'):.2f}")
                
            except Exception as e:
                print(f"Failed to benchmark {run_id}: {e}")
                import traceback
                traceback.print_exc()
            
    # Plot Results
    plt.figure(figsize=(15, 10))
    
    # Use different colors/styles for stocks or models?
    # Let's group by stock in the legend
    
    for run_id, mse_curve in results.items():
        steps = list(mse_curve.keys())
        mses = list(mse_curve.values())
        plt.plot(steps, mses, marker='o', label=run_id)
        
    plt.title(f'Model Performance Degradation over {horizon} Days (Recursive Prediction)')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'benchmark_results_all_models.png'))
    plt.savefig(output_file)
    print(f"\nBenchmark results saved to {output_file}")

if __name__ == "__main__":
    run_benchmark()
