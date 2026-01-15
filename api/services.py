"""
Business logic and services for the API
"""
import os
import sys
import pickle
import uuid
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append('/home/abel/personnal_projects/CAC40_stock_prediction/')

from models.model_lstm_v2 import model_lstm_v2
from api.models import (
    TrainingConfig, TrainingStatus, SimulationRequest, SimulationStatus,
    Transaction, TradingStrategy
)
from api.utils import get_closing_prices, get_strategy_executor
from api.plotting import generate_simulation_chart


class TrainingJobManager:
    """Manages training jobs and their statuses"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingStatus] = {}
        self.models: Dict[str, Any] = {}  # Store trained models
        
    def create_job(self, config: TrainingConfig) -> str:
        """Create a new training job"""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = TrainingStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            current_step="Initializing",
            start_time=datetime.now()
        )
        return job_id
    
    def update_job(self, job_id: str, status: str = None, progress: float = None, 
                   current_step: str = None, error: str = None, model_path: str = None):
        """Update job status"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        if status:
            job.status = status
        if progress is not None:
            job.progress = progress
        if current_step:
            job.current_step = current_step
        if error:
            job.error = error
        if model_path:
            job.model_path = model_path
        
        if status in ["completed", "failed"]:
            job.end_time = datetime.now()
    
    def get_job(self, job_id: str) -> Optional[TrainingStatus]:
        """Get job status"""
        return self.jobs.get(job_id)
    
    def store_model(self, job_id: str, model):
        """Store trained model in memory"""
        self.models[job_id] = model
    
    def get_model(self, job_id: str):
        """Retrieve trained model"""
        return self.models.get(job_id)


class SimulationJobManager:
    """Manages simulation jobs and their statuses"""
    
    def __init__(self):
        self.simulations: Dict[str, SimulationStatus] = {}
        self.transactions: Dict[str, List[Transaction]] = {}
        self.daily_results: Dict[str, List[Dict]] = {}
        
    def create_simulation(self, request: SimulationRequest) -> str:
        """Create a new simulation job"""
        sim_id = str(uuid.uuid4())
        
        from_date_obj = datetime.strptime(request.from_date, '%Y-%m-%d')
        to_date_obj = datetime.strptime(request.to_date, '%Y-%m-%d')
        total_days = (to_date_obj - from_date_obj).days + 1
        
        self.simulations[sim_id] = SimulationStatus(
            sim_id=sim_id,
            status="pending",
            progress=0.0,
            current_date=None,
            days_processed=0,
            total_days=total_days,
            current_balance=request.initial_balance,
            current_stocks=0.0,
            current_stock_value=0.0,
            current_price=0.0,
            total_transactions=0,
            start_time=datetime.now()
        )
        self.transactions[sim_id] = []
        self.daily_results[sim_id] = []
        return sim_id
    
    def update_simulation(self, sim_id: str, **kwargs):
        """Update simulation status"""
        if sim_id not in self.simulations:
            raise ValueError(f"Simulation {sim_id} not found")
        
        sim = self.simulations[sim_id]
        for key, value in kwargs.items():
            if hasattr(sim, key):
                setattr(sim, key, value)
        
        if kwargs.get('status') in ["completed", "failed"]:
            sim.end_time = datetime.now()
    
    def add_transaction(self, sim_id: str, transaction: Transaction):
        """Add a transaction to simulation"""
        if sim_id not in self.transactions:
            self.transactions[sim_id] = []
        self.transactions[sim_id].append(transaction)
        
        # Update transaction count
        if sim_id in self.simulations:
            self.simulations[sim_id].total_transactions = len(self.transactions[sim_id])
    
    def add_daily_result(self, sim_id: str, result: Dict):
        """Add daily result to simulation"""
        if sim_id not in self.daily_results:
            self.daily_results[sim_id] = []
        self.daily_results[sim_id].append(result)
    
    def get_simulation(self, sim_id: str) -> Optional[SimulationStatus]:
        """Get simulation status"""
        return self.simulations.get(sim_id)
    
    def get_transactions(self, sim_id: str) -> List[Transaction]:
        """Get all transactions for a simulation"""
        return self.transactions.get(sim_id, [])
    
    def get_daily_results(self, sim_id: str) -> List[Dict]:
        """Get daily results for a simulation"""
        return self.daily_results.get(sim_id, [])


# Global job manager instances
job_manager = TrainingJobManager()
simulation_manager = SimulationJobManager()


async def train_model_async(job_id: str, config: TrainingConfig):
    """
    Train a model asynchronously with progress updates
    """
    try:
        job_manager.update_job(job_id, status="running", progress=0.1, 
                              current_step="Fetching stock data")
        
        # Create output directory
        model_name = f"from_{config.from_date}_to_{config.to_date}_api_model"
        output_dir = f'/home/abel/personnal_projects/CAC40_stock_prediction/api_models/{model_name}/'
        os.makedirs(output_dir, exist_ok=True)
        
        model_file_path = f'{output_dir}{config.stock_name}_{config.from_date}_{config.to_date}_model.pkl'
        
        # Check if model already exists
        model_loaded = False
        if os.path.exists(model_file_path):
            job_manager.update_job(job_id, progress=0.3, 
                                  current_step="Loading existing model")
            
            try:
                with open(model_file_path, 'rb') as file:
                    model_instance = pickle.load(file)
                
                job_manager.update_job(job_id, progress=1.0, 
                                      current_step="Model loaded successfully",
                                      status="completed",
                                      model_path=model_file_path)
                model_loaded = True
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Corrupted model file found at {model_file_path}. Deleting and retraining. Error: {e}")
                os.remove(model_file_path)
                model_loaded = False

        if not model_loaded:
            # Train new model
            job_manager.update_job(job_id, progress=0.2, 
                                  current_step="Downloading data from Yahoo Finance")
            
            def training_progress_callback(p):
                # Map 0..1 to 0.2..0.9
                real_p = 0.2 + p * 0.7
                job_manager.update_job(job_id, progress=real_p, current_step="Training model...")

            # Load hyperparameters if requested
            hyperparameters = None
            if config.use_stored_hyperparameters:
                try:
                    hp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hyperparameters/best_hyperparameters.json")
                    if os.path.exists(hp_path):
                        with open(hp_path, "r") as f:
                            all_params = json.load(f)
                            if config.stock_name in all_params and "model_lstm_v2" in all_params[config.stock_name]:
                                hyperparameters = all_params[config.stock_name]["model_lstm_v2"]
                                job_manager.update_job(job_id, current_step="Loaded stored hyperparameters")
                except Exception as e:
                    print(f"Could not load hyperparameters: {e}")

            # Run training in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model_instance = await loop.run_in_executor(
                None,
                lambda: model_lstm_v2(
                    model_name,
                    config.stock_name,
                    config.from_date,
                    config.to_date,
                    train_size_percent=config.train_size_percent,
                    val_size_percent=config.val_size_percent,
                    time_step_train_split=config.time_step,
                    global_tuning=config.global_tuning,
                    verbose=False,
                    progress_callback=training_progress_callback,
                    hyperparameters=hyperparameters
                )
            )
            
            job_manager.update_job(job_id, progress=0.9, 
                                  current_step="Saving model")
            
            # Remove callback before pickling to avoid AttributeError
            model_instance.progress_callback = None
            
            # Save model
            with open(model_file_path, 'wb') as file:
                pickle.dump(model_instance, file)
            
            job_manager.update_job(job_id, progress=1.0, 
                                  current_step="Training completed successfully",
                                  status="completed",
                                  model_path=model_file_path)
        
        # Store model in memory
        job_manager.store_model(job_id, model_instance)
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        job_manager.update_job(job_id, status="failed", error=error_msg)
        raise


def make_predictions(job_id: str, n_days: int = 1) -> Dict:
    """
    Make predictions for the next n days using a trained model
    """
    model_instance = job_manager.get_model(job_id)
    if not model_instance:
        raise ValueError(f"No trained model found for job {job_id}")
    
    predictions = []
    current_sequence = model_instance.X[-1].copy()
    
    for day in range(1, n_days + 1):
        # Predict next day
        next_prediction = model_instance.best_model.predict(current_sequence.reshape(1, -1, 1))
        predicted_price = model_instance.scaler.inverse_transform(next_prediction).flatten()[0]
        
        predictions.append({
            "day": day,
            "predicted_price": float(predicted_price)
        })
        
        # Update sequence for next prediction
        if day < n_days:
            current_sequence = np.append(current_sequence[1:], next_prediction[0])
    
    # Get last actual price
    last_actual_price = model_instance.scaler.inverse_transform(
        model_instance.y[-1].reshape(-1, 1)
    ).flatten()[0]
    
    last_idx = model_instance.stock_data.index[-1]
    if isinstance(last_idx, str):
        last_date = last_idx
    else:
        last_date = last_idx.strftime('%Y-%m-%d')
    
    return {
        "predictions": predictions,
        "last_actual_price": float(last_actual_price),
        "last_actual_date": last_date
    }


async def run_historical_simulation(sim_id: str, request: SimulationRequest):
    """
    Run a historical simulation to compare predictions vs actual prices
    Supports multiple strategies and retraining frequency
    """
    try:
        from_date_obj = datetime.strptime(request.from_date, '%Y-%m-%d')
        to_date_obj = datetime.strptime(request.to_date, '%Y-%m-%d')
        
        # Calculate number of days
        nb_days = (to_date_obj - from_date_obj).days
        if nb_days <= 0:
            raise ValueError("to_date must be after from_date")
        
        # Update simulation status
        simulation_manager.update_simulation(
            sim_id,
            status="running",
            progress=0.0,
            total_days=nb_days + 1
        )
        
        # Initialize trading strategies
        strategy_params = {}
        if request.buy_threshold is not None:
            strategy_params['buy_threshold'] = request.buy_threshold
        if request.sell_threshold is not None:
            strategy_params['sell_threshold'] = request.sell_threshold
        if request.min_profit_percentage is not None:
            strategy_params['min_profit_percentage'] = request.min_profit_percentage
        if request.max_loss_percentage is not None:
            strategy_params['max_loss_percentage'] = request.max_loss_percentage
        
        # Determine strategies to run
        strategies_to_run = request.strategies
        if not strategies_to_run and request.strategy:
            strategies_to_run = [request.strategy]
        elif not strategies_to_run:
            strategies_to_run = [TradingStrategy.SIMPLE]
            
        executors = {}
        for strat in strategies_to_run:
            executors[strat.value] = get_strategy_executor(
                strat,
                request.initial_balance,
                strategy_name=strat.value,
                **strategy_params
            )
        
        # Track stats per strategy
        strategy_stats = {name: {"wins": 0, "losses": 0} for name in executors}
        
        model_instance = None
        last_trained_date = None
        
        for day_offset in range(nb_days, -1, -1):
            current_date = to_date_obj - timedelta(days=day_offset)
            training_end_date = current_date.strftime('%Y-%m-%d')
            training_start_date = (current_date - timedelta(days=365 * request.nb_years_data)).strftime('%Y-%m-%d')
            
            # Update progress
            days_processed = nb_days - day_offset
            progress = days_processed / (nb_days + 1)
            
            # Use first executor for status updates (approximate)
            first_executor = list(executors.values())[0]
            simulation_manager.update_simulation(
                sim_id,
                progress=progress,
                current_date=training_end_date,
                days_processed=days_processed,
                current_balance=first_executor.balance,
                current_stocks=first_executor.stocks_owned
            )
            
            try:
                # Determine if we need to retrain or load a new model
                should_retrain = (days_processed % request.retrain_interval == 0) or (model_instance is None)
                
                if should_retrain:
                    # Create/load model for this date
                    model_name = f"sim_{training_start_date}_to_{training_end_date}"
                    output_dir = f'/home/abel/personnal_projects/CAC40_stock_prediction/api_simulations/{model_name}/'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    model_file_path = f'{output_dir}{request.stock_name}_model.pkl'
                    
                    if os.path.exists(model_file_path):
                        with open(model_file_path, 'rb') as file:
                            model_instance = pickle.load(file)
                    else:
                        # Define progress callback for this iteration
                        def sim_training_progress(p):
                            global_p = (days_processed + p) / (nb_days + 1)
                            simulation_manager.update_simulation(
                                sim_id,
                                progress=global_p,
                                current_date=training_end_date,
                                days_processed=days_processed
                            )

                        # Load hyperparameters if requested
                        hyperparameters = None
                        if request.use_stored_hyperparameters:
                            try:
                                hp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hyperparameters/best_hyperparameters.json")
                                if os.path.exists(hp_path):
                                    with open(hp_path, "r") as f:
                                        all_params = json.load(f)
                                        if request.stock_name in all_params and "model_lstm_v2" in all_params[request.stock_name]:
                                            hyperparameters = all_params[request.stock_name]["model_lstm_v2"]
                            except Exception as e:
                                print(f"Could not load hyperparameters: {e}")

                        # Train model for this period
                        loop = asyncio.get_event_loop()
                        model_instance = await loop.run_in_executor(
                            None,
                            lambda: model_lstm_v2(
                                model_name,
                                request.stock_name,
                                training_start_date,
                                training_end_date,
                                train_size_percent=0.8,
                                val_size_percent=0.2,
                                time_step_train_split=request.time_step,
                                global_tuning=True,
                                verbose=False,
                                progress_callback=sim_training_progress,
                                hyperparameters=hyperparameters
                            )
                        )
                        
                        # Remove callback before pickling
                        model_instance.progress_callback = None
                        
                        with open(model_file_path, 'wb') as file:
                            pickle.dump(model_instance, file)
                    
                    last_trained_date = training_end_date
                
                # Make predictions using the current model
                # Note: If we didn't retrain, we are using the model trained on 'last_trained_date'
                # to predict for 'training_end_date'.
                # We need to fetch the sequence ending at 'training_end_date' (yesterday) to predict today.
                # But model_instance.X[-1] is the sequence ending at 'last_trained_date'.
                # So if we are not retraining, we need to construct the input sequence from fresh data.
                
                if not should_retrain:
                    # We need to fetch the latest data up to training_end_date to build the sequence
                    # This is complex because we need to scale it using the model's scaler
                    # For simplicity in this version, we might just rely on the fact that we are simulating
                    # and maybe we can't easily "fast forward" the model without reloading data.
                    # BUT: The user wants to simulate "retraining every N days".
                    # This implies between retrains, we use the OLD model to predict NEW days.
                    # To do this properly, we need the recent price history.
                    
                    # Fetch recent history (last time_step days)
                    recent_start = (current_date - timedelta(days=request.time_step * 2)).strftime('%Y-%m-%d')
                    recent_data = get_closing_prices(recent_start, training_end_date, request.stock_name)
                    
                    if len(recent_data) < request.time_step:
                         raise ValueError(f"Not enough data to predict for {training_end_date}")
                         
                    # Scale data
                    recent_scaled = model_instance.scaler.transform(recent_data.values)
                    
                    # Get last sequence
                    current_sequence = recent_scaled[-request.time_step:].reshape(1, -1, 1)
                    
                    # Predict
                    next_day_predict = model_instance.best_model.predict(current_sequence)
                    predicted_price_dollar = model_instance.scaler.inverse_transform(next_day_predict).flatten()[0]
                    
                    # We also need actual price for today
                    actual_price_dollar = recent_data.iloc[-1]['Close']
                    
                    # And "predicted_last_day" is not really available/relevant if we didn't run for yesterday
                    # We can just use the previous prediction if we stored it, or ignore it.
                    predicted_last_day_dollar = predicted_price_dollar # Placeholder
                    
                else:
                    # Standard case: we just trained/loaded model up to yesterday
                    all_for_next_day = np.append(model_instance.X[-1][1:], model_instance.y[-1])
                    predicted_last_day = model_instance.best_model.predict(model_instance.X[-1].reshape(1, -1, 1))
                    next_day_predict = model_instance.best_model.predict(all_for_next_day.reshape(1, -1, 1))
                    
                    predicted_last_day_dollar = model_instance.scaler.inverse_transform(predicted_last_day).flatten()[0]
                    actual_price_dollar = model_instance.scaler.inverse_transform(model_instance.y[-1].reshape(-1, 1)).flatten()[0]
                    predicted_price_dollar = model_instance.scaler.inverse_transform(next_day_predict).flatten()[0]

                
                # Execute trading strategies
                daily_result = {
                    "date": training_end_date,
                    "actual_price": float(actual_price_dollar),
                    "predicted_price": float(predicted_price_dollar),
                    "predicted_last_day": float(predicted_last_day_dollar),
                    "strategies": {}
                }
                
                for name, executor in executors.items():
                    # Ensure inputs are python floats, not numpy types
                    action, transaction, decision_details = executor.execute_trade(
                        date=training_end_date,
                        actual_price=float(actual_price_dollar),
                        predicted_price=float(predicted_price_dollar),
                        predicted_last_day=float(predicted_last_day_dollar)
                    )
                    
                    # Track transaction
                    if transaction:
                        simulation_manager.add_transaction(sim_id, transaction)
                        
                        # Track wins/losses
                        if action.value == "sell" and executor.last_buy_price:
                            if actual_price_dollar > executor.last_buy_price:
                                strategy_stats[name]["wins"] += 1
                            else:
                                strategy_stats[name]["losses"] += 1
                    
                    # Add strategy result
                    daily_result["strategies"][name] = {
                        "action": action.value,
                        "balance": float(executor.balance),
                        "stocks_owned": float(executor.stocks_owned),
                        "portfolio_value": float(executor.get_portfolio_value(actual_price_dollar)),
                        "decision_details": decision_details
                    }
                
                simulation_manager.add_daily_result(sim_id, daily_result)
                
                # Update simulation status (using first strategy for summary)
                first_strat_name = list(executors.keys())[0]
                first_strat_res = daily_result["strategies"][first_strat_name]
                
                simulation_manager.update_simulation(
                    sim_id,
                    current_balance=first_strat_res["balance"],
                    current_stocks=first_strat_res["stocks_owned"],
                    current_stock_value=first_strat_res["stocks_owned"] * actual_price_dollar,
                    current_price=float(actual_price_dollar)
                )
                
            except Exception as e:
                # Log error but continue simulation
                error_result = {
                    "date": training_end_date,
                    "error": str(e)
                }
                simulation_manager.add_daily_result(sim_id, error_result)
        
        # Calculate final results
        daily_results = simulation_manager.get_daily_results(sim_id)
        transactions = simulation_manager.get_transactions(sim_id)
        
        # Find last valid price
        final_actual_price = 0.0
        for result in reversed(daily_results):
            if "actual_price" in result:
                final_actual_price = result["actual_price"]
                break
        
        strategies_results = {}
        for name, executor in executors.items():
            final_stock_value = executor.stocks_owned * final_actual_price
            final_balance = executor.balance + final_stock_value
            benefit = final_balance - request.initial_balance
            benefit_percentage = (benefit / request.initial_balance) * 100
            
            buy_trades = sum(1 for t in transactions if t.strategy == name and t.transaction_type.value == "buy")
            sell_trades = sum(1 for t in transactions if t.strategy == name and t.transaction_type.value == "sell")
            
            strategies_results[name] = {
                "final_balance": final_balance,
                "final_stocks_owned": executor.stocks_owned,
                "final_stock_value": final_stock_value,
                "benefit": benefit,
                "benefit_percentage": benefit_percentage,
                "total_trades": buy_trades + sell_trades,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "wins": strategy_stats[name]["wins"],
                "losses": strategy_stats[name]["losses"]
            }
            
        # Use first strategy for top-level backward compatibility
        first_name = list(executors.keys())[0]
        first_res = strategies_results[first_name]
        
        # Generate simulation chart (using first strategy for now, or maybe all?)
        # For now, let's stick to the first one to avoid breaking the plotter
        plot_path = generate_simulation_chart(sim_id, daily_results, request.stock_name)
        
        # Mark simulation as completed
        simulation_manager.update_simulation(
            sim_id,
            status="completed",
            progress=1.0,
            current_balance=executors[first_name].balance,
            current_stocks=executors[first_name].stocks_owned,
            current_stock_value=executors[first_name].stocks_owned * final_actual_price,
            plot_path=plot_path
        )
        
        return {
            "sim_id": sim_id,
            "status": "completed",
            "stock_name": request.stock_name,
            "plot_path": plot_path,
            "simulation_period": {
                "from": request.from_date,
                "to": request.to_date
            },
            "initial_balance": request.initial_balance,
            "final_balance": first_res["final_balance"],
            "final_stocks_owned": first_res["final_stocks_owned"],
            "final_stock_value": first_res["final_stock_value"],
            "benefit": first_res["benefit"],
            "benefit_percentage": first_res["benefit_percentage"],
            "strategy_used": first_name,
            "strategies_results": strategies_results,
            "daily_results": daily_results,
            "transactions": transactions,
            "summary": {
                "total_trades": first_res["total_trades"],
                "buy_trades": first_res["buy_trades"],
                "sell_trades": first_res["sell_trades"],
                "winning_trades": first_res["wins"],
                "losing_trades": first_res["losses"],
                "win_rate": (first_res["wins"] / first_res["sell_trades"] * 100) if first_res["sell_trades"] > 0 else 0,
                "total_days": len(daily_results),
                "days_with_errors": sum(1 for d in daily_results if "error" in d)
            }
        }
        
    except Exception as e:
        # Mark simulation as failed
        simulation_manager.update_simulation(
            sim_id,
            status="failed",
            error=str(e)
        )
        raise
