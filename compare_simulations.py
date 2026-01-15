import asyncio
import time
import os
import shutil
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.getcwd())

from api.models import SimulationRequest
from api.services import simulation_manager, run_historical_simulation

async def run_comparison():
    stock_name = "ENGI.PA"
    # Use a short period for demonstration, e.g., 3 days
    # Ensure these dates are within the range of available data and future enough to trigger the simulation logic
    # The workspace has data up to 2025-11-10 based on file names in dataset/
    # Let's use a range that likely has data.
    from_date = "2025-11-01"
    to_date = "2025-11-10" 
    
    print(f"--- Starting Comparison for {stock_name} from {from_date} to {to_date} ---")

    # Helper to clean up model files to force re-training
    def cleanup_models():
        print("Cleaning up cached simulation models...")
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
        delta = end - start
        for i in range(delta.days + 1):
            current = end - timedelta(days=i)
            # Logic from services.py to find the folder
            # training_end_date = current.strftime('%Y-%m-%d')
            # training_start_date = (current - timedelta(days=365 * 2)).strftime('%Y-%m-%d') # Default nb_years_data=2
            # model_name = f"sim_{training_start_date}_to_{training_end_date}"
            # But wait, the folder name depends on the exact start date which depends on nb_years_data.
            # Let's just look for folders starting with sim_ and containing the end date?
            # Or better, just delete the specific folders if we can calculate them.
            
            # Replicating logic from services.py
            training_end_date = current.strftime('%Y-%m-%d')
            training_start_date = (current - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
            model_name = f"sim_{training_start_date}_to_{training_end_date}"
            output_dir = f'api_simulations/{model_name}/'
            
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                # print(f"Deleted {output_dir}")

    # 1. Run WITH stored hyperparameters (Fine-tuning disabled/cached) - FAST
    cleanup_models()
    print("\n[1/2] Running simulation WITHOUT fine-tuning (use_stored_hyperparameters=True)...")
    req_cached = SimulationRequest(
        stock_name=stock_name,
        from_date=from_date,
        to_date=to_date,
        initial_balance=10000,
        strategy="conservative",
        nb_years_data=2,
        time_step=60,
        use_stored_hyperparameters=True
    )
    
    sim_id_cached = simulation_manager.create_simulation(req_cached)
    
    start_time = time.time()
    try:
        await run_historical_simulation(sim_id_cached, req_cached)
    except Exception as e:
        print(f"Error during cached simulation: {e}")
    end_time = time.time()
    duration_cached = end_time - start_time
    print(f"Duration WITHOUT fine-tuning: {duration_cached:.2f} seconds")

    # 2. Run WITHOUT stored hyperparameters (Fine-tuning enabled) - SLOW
    cleanup_models()
    print("\n[2/2] Running simulation WITH fine-tuning (use_stored_hyperparameters=False)...")
    req_tuning = SimulationRequest(
        stock_name=stock_name,
        from_date=from_date,
        to_date=to_date,
        initial_balance=10000,
        strategy="conservative",
        nb_years_data=2,
        time_step=60,
        use_stored_hyperparameters=False
    )
    
    sim_id_tuning = simulation_manager.create_simulation(req_tuning)
    
    start_time = time.time()
    try:
        await run_historical_simulation(sim_id_tuning, req_tuning)
    except Exception as e:
        print(f"Error during tuning simulation: {e}")
    end_time = time.time()
    duration_tuning = end_time - start_time
    print(f"Duration WITH fine-tuning: {duration_tuning:.2f} seconds")

    # Comparison
    print("\n--- Results ---")
    print(f"With Fine-tuning:    {duration_tuning:.2f}s")
    print(f"With Stored Params:  {duration_cached:.2f}s")
    if duration_cached < duration_tuning:
        speedup = duration_tuning / duration_cached
        print(f"Speedup: {speedup:.2f}x faster")
    else:
        print("No speedup observed (check if hyperparameters were actually loaded or if tuning is fast).")

if __name__ == "__main__":
    asyncio.run(run_comparison())
