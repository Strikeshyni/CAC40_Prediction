"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum


class TradingStrategy(str, Enum):
    """Trading strategy types"""
    SIMPLE = "simple"  # Buy if predicted > actual, sell if predicted < actual
    THRESHOLD = "threshold"  # Buy/sell only if difference exceeds threshold
    PERCENTAGE = "percentage"  # Buy/sell based on percentage change
    CONSERVATIVE = "conservative"  # Only trade with high confidence
    AGGRESSIVE = "aggressive"  # Trade on small signals


class TransactionType(str, Enum):
    """Transaction types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class TrainingConfig(BaseModel):
    """Configuration for model training"""
    stock_name: str = Field(..., description="Stock symbol (e.g., ENGI.PA)")
    from_date: str = Field(..., description="Start date for training data (YYYY-MM-DD)")
    to_date: str = Field(..., description="End date for training data (YYYY-MM-DD)")
    train_size_percent: float = Field(0.8, ge=0.1, le=0.95, description="Training data percentage")
    val_size_percent: float = Field(0.2, ge=0.05, le=0.9, description="Validation data percentage")
    time_step: int = Field(300, ge=10, le=1000, description="Time steps for sequence prediction")
    global_tuning: bool = Field(True, description="Enable hyperparameter tuning")
    use_stored_hyperparameters: bool = Field(False, description="Use pre-calculated hyperparameters if available")
    
    @validator('from_date', 'to_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @validator('to_date')
    def validate_date_range(cls, v, values):
        if 'from_date' in values:
            from_d = datetime.strptime(values['from_date'], '%Y-%m-%d')
            to_d = datetime.strptime(v, '%Y-%m-%d')
            if to_d <= from_d:
                raise ValueError('to_date must be after from_date')
        return v
    
    @validator('train_size_percent')
    def validate_train_val_sum(cls, v, values):
        if 'val_size_percent' in values:
            if v + values['val_size_percent'] > 1.0:
                raise ValueError('train_size_percent + val_size_percent must not exceed 1.0')
        return v


class TrainingResponse(BaseModel):
    """Response after starting a training job"""
    job_id: str
    status: str
    message: str
    config: TrainingConfig


class TrainingStatus(BaseModel):
    """Status of a training job"""
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    current_step: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    model_path: Optional[str] = None


class PredictionRequest(BaseModel):
    """Request for making predictions"""
    job_id: str
    n_days: int = Field(1, ge=1, le=30, description="Number of days to predict")


class PredictionResponse(BaseModel):
    """Response with predictions"""
    job_id: str
    stock_name: str
    predictions: List[Dict[str, float]]  # [{"day": 1, "predicted_price": 123.45}, ...]
    last_actual_price: float
    last_actual_date: str


class SimulationRequest(BaseModel):
    """Request for historical simulation"""
    stock_name: str
    from_date: str
    to_date: str
    initial_balance: float = Field(100.0, gt=0, description="Initial cash balance")
    time_step: int = Field(300, ge=10, le=1000)
    nb_years_data: int = Field(10, ge=1, le=20, description="Years of historical data to use")
    
    # Trading strategy parameters
    strategy: Optional[TradingStrategy] = Field(None, description="Primary trading strategy (deprecated, use strategies)")
    strategies: List[TradingStrategy] = Field(default_factory=lambda: [TradingStrategy.SIMPLE], description="List of strategies to compare")
    retrain_interval: int = Field(1, ge=1, le=365, description="Retrain model every N days")
    
    buy_threshold: Optional[float] = Field(None, ge=0, description="Minimum difference to trigger buy (euros or %)")
    sell_threshold: Optional[float] = Field(None, ge=0, description="Minimum difference to trigger sell (euros or %)")
    min_profit_percentage: Optional[float] = Field(None, ge=0, le=100, description="Minimum profit % before selling")
    max_loss_percentage: Optional[float] = Field(None, ge=0, le=100, description="Maximum loss % before stop-loss")
    confidence_threshold: Optional[float] = Field(None, ge=0, le=1, description="Minimum prediction confidence (0-1)")
    use_stored_hyperparameters: bool = Field(False, description="Use pre-calculated hyperparameters if available")
    
    @validator('from_date', 'to_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')



class Transaction(BaseModel):
    """Detailed transaction record"""
    transaction_id: int
    strategy: str = "default"
    date: str
    transaction_type: TransactionType
    stock_price: float
    quantity: float
    total_value: float
    balance_after: float
    stocks_owned_after: float
    reason: str
    predicted_price: Optional[float] = None
    predicted_change_pct: Optional[float] = None


class SimulationStatus(BaseModel):
    """Status of a running simulation"""
    sim_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    current_date: Optional[str] = None
    days_processed: int
    total_days: int
    current_balance: float
    current_stocks: float
    current_stock_value: Optional[float] = None
    current_price: Optional[float] = None
    total_transactions: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    plot_path: Optional[str] = None


class SimulationResponse(BaseModel):
    """Response after starting or completing a simulation"""
    sim_id: str
    status: str
    stock_name: str
    simulation_period: Dict[str, str]  # {"from": "...", "to": "..."}
    initial_balance: float
    final_balance: Optional[float] = None
    final_stocks_owned: Optional[float] = None
    final_stock_value: Optional[float] = None
    benefit: Optional[float] = None
    benefit_percentage: Optional[float] = None
    strategy_used: Optional[str] = None
    strategies_results: Optional[Dict[str, Dict[str, Any]]] = None
    daily_results: Optional[List[Dict[str, Any]]] = None
    transactions: Optional[List[Transaction]] = None
    summary: Optional[Dict[str, Any]] = None


class TransactionsResponse(BaseModel):
    """Response with transaction history"""
    sim_id: str
    total_transactions: int
    transactions: List[Transaction]
    summary: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
