"""
Business logic and services for the API
"""
import os
import sys
import pickle
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List
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


class TrainingJobManager:
    """Manages training jobs and their statuses"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingStatus] = {}
        self.models: Dict[str, any] = {}  # Store trained models
        
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
        if os.path.exists(model_file_path):
            job_manager.update_job(job_id, progress=0.3, 
                                  current_step="Loading existing model")
            
            with open(model_file_path, 'rb') as file:
                model_instance = pickle.load(file)
                
            job_manager.update_job(job_id, progress=1.0, 
                                  current_step="Model loaded successfully",
                                  status="completed",
                                  model_path=model_file_path)
        else:
            # Train new model
            job_manager.update_job(job_id, progress=0.2, 
                                  current_step="Downloading data from Yahoo Finance")
            
            def training_progress_callback(p):
                # Map 0..1 to 0.2..0.9
                real_p = 0.2 + p * 0.7
                job_manager.update_job(job_id, progress=real_p, current_step="Training model...")

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
                    progress_callback=training_progress_callback
                )
            )
            
            job_manager.update_job(job_id, progress=0.9, 
                                  current_step="Saving model")
            
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
    
    last_date = model_instance.stock_data.index[-1].strftime('%Y-%m-%d')
    
    return {
        "predictions": predictions,
        "last_actual_price": float(last_actual_price),
        "last_actual_date": last_date
    }


async def run_historical_simulation(sim_id: str, request: SimulationRequest):
    """
    Run a historical simulation to compare predictions vs actual prices
    Now uses the new strategy system and tracks all transactions
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
        
        # Initialize trading strategy
        strategy_params = {}
        if request.buy_threshold is not None:
            strategy_params['buy_threshold'] = request.buy_threshold
        if request.sell_threshold is not None:
            strategy_params['sell_threshold'] = request.sell_threshold
        if request.min_profit_percentage is not None:
            strategy_params['min_profit_percentage'] = request.min_profit_percentage
        if request.max_loss_percentage is not None:
            strategy_params['max_loss_percentage'] = request.max_loss_percentage
        
        strategy_executor = get_strategy_executor(
            request.strategy,
            request.initial_balance,
            **strategy_params
        )
        
        wins = 0
        losses = 0
        
        for day_offset in range(nb_days, -1, -1):
            current_date = to_date_obj - timedelta(days=day_offset)
            training_end_date = current_date.strftime('%Y-%m-%d')
            training_start_date = (current_date - timedelta(days=365 * request.nb_years_data)).strftime('%Y-%m-%d')
            
            # Update progress
            days_processed = nb_days - day_offset
            progress = days_processed / (nb_days + 1)
            simulation_manager.update_simulation(
                sim_id,
                progress=progress,
                current_date=training_end_date,
                days_processed=days_processed,
                current_balance=strategy_executor.balance,
                current_stocks=strategy_executor.stocks_owned
            )
            
            try:
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
                        # Global progress = (days_processed + p) / (nb_days + 1)
                        # We allocate the full "step" to training for simplicity in visualization
                        global_p = (days_processed + p) / (nb_days + 1)
                        simulation_manager.update_simulation(
                            sim_id,
                            progress=global_p,
                            current_date=training_end_date,
                            days_processed=days_processed,
                            current_balance=strategy_executor.balance,
                            current_stocks=strategy_executor.stocks_owned
                        )

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
                            progress_callback=sim_training_progress
                        )
                    )
                    
                    with open(model_file_path, 'wb') as file:
                        pickle.dump(model_instance, file)
                
                # Make predictions
                all_for_next_day = np.append(model_instance.X[-1][1:], model_instance.y[-1])
                predicted_last_day = model_instance.best_model.predict(model_instance.X[-1].reshape(1, -1, 1))
                next_day_predict = model_instance.best_model.predict(all_for_next_day.reshape(1, -1, 1))
                
                predicted_last_day_dollar = model_instance.scaler.inverse_transform(predicted_last_day).flatten()[0]
                actual_price_dollar = model_instance.scaler.inverse_transform(model_instance.y[-1].reshape(-1, 1)).flatten()[0]
                predicted_price_dollar = model_instance.scaler.inverse_transform(next_day_predict).flatten()[0]
                
                # Execute trading strategy
                action, transaction = strategy_executor.execute_trade(
                    date=training_end_date,
                    actual_price=actual_price_dollar,
                    predicted_price=predicted_price_dollar,
                    predicted_last_day=predicted_last_day_dollar
                )
                
                # Track transaction
                if transaction:
                    simulation_manager.add_transaction(sim_id, transaction)
                    
                    # Track wins/losses
                    if action.value == "sell" and strategy_executor.last_buy_price:
                        if actual_price_dollar > strategy_executor.last_buy_price:
                            wins += 1
                        else:
                            losses += 1
                
                # Add daily result
                daily_result = {
                    "date": training_end_date,
                    "actual_price": float(actual_price_dollar),
                    "predicted_price": float(predicted_price_dollar),
                    "predicted_last_day": float(predicted_last_day_dollar),
                    "action": action.value,
                    "balance": float(strategy_executor.balance),
                    "stocks_owned": float(strategy_executor.stocks_owned),
                    "portfolio_value": float(strategy_executor.get_portfolio_value(actual_price_dollar))
                }
                simulation_manager.add_daily_result(sim_id, daily_result)
                
            except Exception as e:
                # Log error but continue simulation
                error_result = {
                    "date": training_end_date,
                    "error": str(e)
                }
                simulation_manager.add_daily_result(sim_id, error_result)
        
        # Calculate final balance
        daily_results = simulation_manager.get_daily_results(sim_id)
        transactions = simulation_manager.get_transactions(sim_id)
        
        if daily_results and "actual_price" in daily_results[-1]:
            final_actual_price = daily_results[-1]["actual_price"]
            final_balance = strategy_executor.balance + strategy_executor.stocks_owned * final_actual_price
        else:
            final_balance = strategy_executor.balance
        
        benefit = final_balance - request.initial_balance
        benefit_percentage = (benefit / request.initial_balance) * 100
        
        # Calculate trade statistics
        buy_trades = sum(1 for t in transactions if t.transaction_type.value == "buy")
        sell_trades = sum(1 for t in transactions if t.transaction_type.value == "sell")
        
        # Mark simulation as completed
        simulation_manager.update_simulation(
            sim_id,
            status="completed",
            progress=1.0,
            current_balance=strategy_executor.balance,
            current_stocks=strategy_executor.stocks_owned
        )
        
        return {
            "sim_id": sim_id,
            "status": "completed",
            "stock_name": request.stock_name,
            "simulation_period": {
                "from": request.from_date,
                "to": request.to_date
            },
            "initial_balance": request.initial_balance,
            "final_balance": final_balance,
            "benefit": benefit,
            "benefit_percentage": benefit_percentage,
            "strategy_used": request.strategy,
            "daily_results": daily_results,
            "transactions": transactions,
            "summary": {
                "total_trades": len(transactions),
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate": (wins / sell_trades * 100) if sell_trades > 0 else 0,
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
