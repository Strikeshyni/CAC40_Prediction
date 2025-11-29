"""
FastAPI main application
Stock prediction API for CAC40 stocks
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import asyncio
from typing import List
import json
import os

from api.models import (
    TrainingConfig, TrainingResponse, TrainingStatus,
    PredictionRequest, PredictionResponse,
    SimulationRequest, SimulationResponse, SimulationStatus,
    TransactionsResponse, Transaction,
    ErrorResponse, HealthResponse
)
from api.services import (
    job_manager, simulation_manager, train_model_async, make_predictions, run_historical_simulation
)


# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_job_update(self, job_id: str, status: TrainingStatus):
        """Send job status update to all connected clients"""
        message = {
            "job_id": job_id,
            "status": status.status,
            "progress": status.progress,
            "current_step": status.current_step
        }
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("API starting up...")
    yield
    print("API shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="CAC40 Stock Prediction API",
    description="API for training ML models and predicting stock prices",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(status="healthy")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(status="healthy")


@app.post("/api/train", response_model=TrainingResponse, status_code=202)
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    Start a new model training job
    
    - **stock_name**: Stock symbol (e.g., ENGI.PA)
    - **from_date**: Start date for training data (YYYY-MM-DD)
    - **to_date**: End date for training data (YYYY-MM-DD)
    - **train_size_percent**: Training data percentage (0.1-0.95)
    - **val_size_percent**: Validation data percentage (0.05-0.9)
    - **time_step**: Time steps for sequence prediction (10-1000)
    - **global_tuning**: Enable hyperparameter tuning
    """
    try:
        # Create job
        job_id = job_manager.create_job(config)
        
        # Start training in background
        background_tasks.add_task(train_model_async, job_id, config)
        
        return TrainingResponse(
            job_id=job_id,
            status="pending",
            message="Training job started successfully",
            config=config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/train/{job_id}/status", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """
    Get the status of a training job
    
    - **job_id**: The unique identifier of the training job
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job


@app.get("/api/train/jobs", response_model=List[TrainingStatus])
async def list_all_jobs():
    """
    List all training jobs
    """
    return list(job_manager.jobs.values())


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using a trained model
    
    - **job_id**: The ID of the completed training job
    - **n_days**: Number of days to predict (1-30)
    """
    # Check if job exists and is completed
    job = job_manager.get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {request.job_id} not found")
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job {request.job_id} is not completed yet. Current status: {job.status}"
        )
    
    try:
        result = make_predictions(request.job_id, request.n_days)
        
        # Get stock name from model
        model = job_manager.get_model(request.job_id)
        stock_name = model.stock_name if hasattr(model, 'stock_name') else "unknown"
        
        return PredictionResponse(
            job_id=request.job_id,
            stock_name=stock_name,
            predictions=result["predictions"],
            last_actual_price=result["last_actual_price"],
            last_actual_date=result["last_actual_date"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate", response_model=SimulationResponse, status_code=202)
async def start_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """
    Start a historical simulation to test trading strategy
    
    - **stock_name**: Stock symbol (e.g., ENGI.PA)
    - **from_date**: Start date for simulation (YYYY-MM-DD)
    - **to_date**: End date for simulation (YYYY-MM-DD)
    - **initial_balance**: Initial cash balance
    - **strategy**: Trading strategy to use (simple, threshold, percentage, conservative, aggressive)
    - **buy_threshold**: Threshold for buy signals (meaning depends on strategy)
    - **sell_threshold**: Threshold for sell signals
    - **min_profit_percentage**: Minimum profit % before selling (conservative strategy)
    - **max_loss_percentage**: Maximum loss % before stop-loss (aggressive strategy)
    """
    try:
        # Create simulation
        sim_id = simulation_manager.create_simulation(request)
        
        # Start simulation in background
        background_tasks.add_task(run_historical_simulation, sim_id, request)
        
        return SimulationResponse(
            sim_id=sim_id,
            status="pending",
            stock_name=request.stock_name,
            simulation_period={
                "from": request.from_date,
                "to": request.to_date
            },
            initial_balance=request.initial_balance,
            strategy_used=request.strategy
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulate/{sim_id}/status", response_model=SimulationStatus)
async def get_simulation_status(sim_id: str):
    """
    Get the status of a running simulation
    
    - **sim_id**: The unique identifier of the simulation
    """
    sim = simulation_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    return sim


@app.get("/api/simulate/{sim_id}/transactions", response_model=TransactionsResponse)
async def get_simulation_transactions(sim_id: str):
    """
    Get all transactions for a simulation
    
    - **sim_id**: The unique identifier of the simulation
    """
    sim = simulation_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    transactions = simulation_manager.get_transactions(sim_id)
    
    # Calculate summary
    buy_trades = sum(1 for t in transactions if t.transaction_type.value == "buy")
    sell_trades = sum(1 for t in transactions if t.transaction_type.value == "sell")
    
    total_invested = sum(t.total_value for t in transactions if t.transaction_type.value == "buy")
    total_returned = sum(t.total_value for t in transactions if t.transaction_type.value == "sell")
    
    summary = {
        "total_transactions": len(transactions),
        "buy_transactions": buy_trades,
        "sell_transactions": sell_trades,
        "total_invested": total_invested,
        "total_returned": total_returned,
        "net_trading_result": total_returned - total_invested
    }
    
    return TransactionsResponse(
        sim_id=sim_id,
        total_transactions=len(transactions),
        transactions=transactions,
        summary=summary
    )


@app.get("/api/simulate/{sim_id}/results", response_model=SimulationResponse)
async def get_simulation_results(sim_id: str):
    """
    Get complete results of a simulation (only available when completed)
    
    - **sim_id**: The unique identifier of the simulation
    """
    sim = simulation_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    if sim.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Simulation {sim_id} is not completed yet. Current status: {sim.status}"
        )
    
    daily_results = simulation_manager.get_daily_results(sim_id)
    transactions = simulation_manager.get_transactions(sim_id)
    
    # Calculate final balance from last VALID daily result
    final_balance = sim.current_balance
    final_stock_value = 0.0
    
    # Find last valid result with portfolio value
    last_valid_result = None
    for result in reversed(daily_results):
        if "portfolio_value" in result:
            last_valid_result = result
            break
            
    if last_valid_result:
        if "actual_price" in last_valid_result:
            final_stock_value = sim.current_stocks * last_valid_result["actual_price"]
    else:
        # Fallback: try to find last price
        last_price = 0.0
        for result in reversed(daily_results):
            if "actual_price" in result:
                last_price = result["actual_price"]
                break
        final_stock_value = sim.current_stocks * last_price
        final_balance = sim.current_balance
    
    # Get initial balance from first valid result or simulation start
    initial_balance = 0.0
    for result in daily_results:
        if "portfolio_value" in result:
            initial_balance = result["portfolio_value"]
            break
    if initial_balance == 0:
        initial_balance = sim.current_balance  # Fallback
        
    benefit = final_balance + final_stock_value - initial_balance
    benefit_percentage = (benefit / initial_balance * 100) if initial_balance > 0 else 0
    
    # Calculate trade statistics
    buy_trades = sum(1 for t in transactions if t.transaction_type.value == "buy")
    sell_trades = sum(1 for t in transactions if t.transaction_type.value == "sell")
    
    wins = 0
    losses = 0
    for t in transactions:
        if t.transaction_type.value == "sell" and "Profit:" in t.reason:
            # Extract profit from reason
            if "Profit:" in t.reason:
                profit_str = t.reason.split("Profit:")[1].strip().split("%")[0].strip()
                try:
                    profit_pct = float(profit_str.replace("(", "").replace(")", ""))
                    if profit_pct > 0:
                        wins += 1
                    else:
                        losses += 1
                except:
                    pass
    
    return SimulationResponse(
        sim_id=sim_id,
        status=sim.status,
        stock_name="",  # Would need to store this in simulation
        simulation_period={"from": "", "to": ""},  # Would need to store this
        initial_balance=initial_balance,
        final_balance=final_balance,
        final_stocks_owned=sim.current_stocks,
        final_stock_value=final_stock_value,
        benefit=benefit,
        benefit_percentage=benefit_percentage,
        strategy_used="simple",  # Would need to store this
        daily_results=daily_results,
        transactions=transactions,
        summary={
            "total_trades": len(transactions),
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate": (wins / sell_trades * 100) if sell_trades > 0 else 0,
            "total_days": len(daily_results),
            "days_with_errors": sum(1 for d in daily_results if "error" in d)
        }
    )


@app.get("/api/simulate/jobs", response_model=List[SimulationStatus])
async def list_all_simulations():
    """
    List all simulations
    """
    return list(simulation_manager.simulations.values())


@app.delete("/api/simulate/{sim_id}")
async def delete_simulation(sim_id: str):
    """
    Delete a simulation from memory
    
    - **sim_id**: The unique identifier of the simulation
    """
    sim = simulation_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    # Remove from simulations, transactions, and daily results
    if sim_id in simulation_manager.simulations:
        del simulation_manager.simulations[sim_id]
    if sim_id in simulation_manager.transactions:
        del simulation_manager.transactions[sim_id]
    if sim_id in simulation_manager.daily_results:
        del simulation_manager.daily_results[sim_id]
    
    return {"message": f"Simulation {sim_id} deleted successfully"}


@app.get("/api/simulation/{sim_id}/plot")
async def get_simulation_plot(sim_id: str):
    """
    Get the plot image for a completed simulation
    """
    sim = simulation_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    if not sim.plot_path or not os.path.exists(sim.plot_path):
        raise HTTPException(status_code=404, detail="Plot not found for this simulation")
        
    return FileResponse(sim.plot_path, media_type="image/png")


@app.websocket("/ws/simulation/{sim_id}")
async def websocket_simulation_updates(websocket: WebSocket, sim_id: str):
    """
    WebSocket endpoint for real-time simulation progress updates
    """
    await manager.connect(websocket)
    try:
        while True:
            # Send current simulation status
            sim = simulation_manager.get_simulation(sim_id)
            if sim:
                await websocket.send_json({
                    "sim_id": sim_id,
                    "status": sim.status,
                    "progress": sim.progress,
                    "current_date": sim.current_date,
                    "days_processed": sim.days_processed,
                    "total_days": sim.total_days,
                    "current_balance": sim.current_balance,
                    "current_stocks": sim.current_stocks,
                    "current_stock_value": sim.current_stock_value,
                    "current_price": sim.current_price,
                    "total_transactions": sim.total_transactions,
                    "error": sim.error
                })
                
                # If simulation is completed or failed, close connection
                if sim.status in ["completed", "failed"]:
                    break
            
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/training/{job_id}")
async def websocket_training_updates(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time training progress updates
    """
    await manager.connect(websocket)
    try:
        while True:
            # Send current job status
            job = job_manager.get_job(job_id)
            if job:
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "current_step": job.current_step,
                    "error": job.error
                })
                
                # If job is completed or failed, close connection
                if job.status in ["completed", "failed"]:
                    break
            
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.delete("/api/train/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a training job from memory
    
    - **job_id**: The unique identifier of the training job
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Remove from jobs and models
    if job_id in job_manager.jobs:
        del job_manager.jobs[job_id]
    if job_id in job_manager.models:
        del job_manager.models[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
