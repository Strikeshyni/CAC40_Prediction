"""
Utility functions for the API
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
from api.models import TradingStrategy, TransactionType, Transaction


def get_closing_prices(from_date: str, to_date: str, stock_name: str):
    """
    Fetch closing prices from Yahoo Finance with caching
    """
    file_path = f'/home/abel/personnal_projects/CAC40_stock_prediction/dataset/{stock_name}_closing_prices_{from_date}_to_{to_date}.csv'

    # Check if data exists
    if os.path.exists(file_path):
        stock_data = pd.read_csv(file_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        
        # Verify data is not empty
        if not stock_data.empty and 'Close' in stock_data.columns:
            return stock_data[['Close']]
        else:
            # If cached file is empty, delete it and re-fetch
            os.remove(file_path)

    # Fetch from Yahoo Finance
    data = yf.download(stock_name, start=from_date, end=to_date)
    
    # Check if download was successful
    if data.empty:
        raise ValueError(f"No data retrieved for {stock_name} from {from_date} to {to_date}. "
                        "This might be due to rate limiting or invalid stock symbol.")

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        stock_data = data['Close'].reset_index()
        stock_data.rename(columns={stock_name: 'Close'}, inplace=True)
    else:
        stock_data = data[['Close']].reset_index()

    # Prepare dataset
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    # Only save if data is not empty
    if not stock_data.empty:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        stock_data.to_csv(file_path)

    return stock_data[['Close']]


def validate_stock_symbol(stock_symbol: str) -> bool:
    """
    Validate if a stock symbol exists
    """
    try:
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info
        return 'regularMarketPrice' in info or 'currentPrice' in info
    except:
        return False


def format_date(date_str: str) -> str:
    """
    Validate and format date string
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


class TradingStrategyExecutor:
    """Base class for trading strategy execution"""
    
    def __init__(self, initial_balance: float, strategy_name: str = "default", **kwargs):
        self.balance = initial_balance
        self.stocks_owned = 0.0
        self.last_buy_price = None
        self.transactions = []
        self.transaction_counter = 0
        self.strategy_name = strategy_name
        self.kwargs = kwargs
    
    def execute_trade(
        self, 
        date: str,
        actual_price: float, 
        predicted_price: float,
        predicted_last_day: float
    ) -> Tuple[TransactionType, Optional[Transaction], Dict[str, Any]]:
        """
        Execute trade based on strategy. Returns (action, transaction, decision_details)
        Must be implemented by subclasses
        """
        raise NotImplementedError
    
    def _create_transaction(
        self,
        date: str,
        transaction_type: TransactionType,
        price: float,
        quantity: float,
        reason: str,
        predicted_price: Optional[float] = None
    ) -> Transaction:
        """Create a transaction record"""
        self.transaction_counter += 1
        
        total_value = price * quantity
        predicted_change = None
        if predicted_price and price > 0:
            predicted_change = ((predicted_price - price) / price) * 100
        
        transaction = Transaction(
            transaction_id=self.transaction_counter,
            strategy=self.strategy_name,
            date=date,
            transaction_type=transaction_type,
            stock_price=price,
            quantity=quantity,
            total_value=total_value,
            balance_after=self.balance,
            stocks_owned_after=self.stocks_owned,
            reason=reason,
            predicted_price=predicted_price,
            predicted_change_pct=predicted_change
        )
        
        self.transactions.append(transaction)
        return transaction
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        return self.balance + (self.stocks_owned * current_price)


class SimpleStrategy(TradingStrategyExecutor):
    """Simple strategy: buy if predicted > actual, sell if predicted < actual"""
    
    def execute_trade(
        self, 
        date: str,
        actual_price: float, 
        predicted_price: float,
        predicted_last_day: float
    ) -> Tuple[TransactionType, Optional[Transaction], Dict[str, Any]]:
        
        diff = predicted_price - actual_price
        diff_pct = (diff / actual_price) * 100 if actual_price > 0 else 0
        
        decision_details = {
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "diff": diff,
            "diff_pct": diff_pct,
            "threshold": 0.0
        }

        # Buy signal: predicted price is higher than actual
        if predicted_price > actual_price and self.balance > 0:
            quantity = self.balance / actual_price
            self.stocks_owned += quantity
            self.balance = 0
            self.last_buy_price = actual_price
            
            transaction = self._create_transaction(
                date=date,
                transaction_type=TransactionType.BUY,
                price=actual_price,
                quantity=quantity,
                reason=f"Predicted increase: {predicted_price:.2f} > {actual_price:.2f} (Diff: {diff:.2f}€, {diff_pct:.2f}%)",
                predicted_price=predicted_price
            )
            return TransactionType.BUY, transaction, decision_details
        
        # Sell signal: predicted price is lower than actual and we own stocks
        elif predicted_price < actual_price and self.stocks_owned > 0:
            quantity = self.stocks_owned
            self.balance += quantity * actual_price
            self.stocks_owned = 0
            
            profit_loss = ""
            if self.last_buy_price:
                profit = ((actual_price - self.last_buy_price) / self.last_buy_price) * 100
                profit_loss = f" (Profit: {profit:.2f}%)"
            
            transaction = self._create_transaction(
                date=date,
                transaction_type=TransactionType.SELL,
                price=actual_price,
                quantity=quantity,
                reason=f"Predicted decrease: {predicted_price:.2f} < {actual_price:.2f} (Diff: {diff:.2f}€, {diff_pct:.2f}%){profit_loss}",
                predicted_price=predicted_price
            )
            return TransactionType.SELL, transaction, decision_details
        
        return TransactionType.HOLD, None, decision_details


class ThresholdStrategy(TradingStrategyExecutor):
    """Threshold strategy: only trade if price difference exceeds threshold"""
    
    def execute_trade(
        self, 
        date: str,
        actual_price: float, 
        predicted_price: float,
        predicted_last_day: float
    ) -> Tuple[TransactionType, Optional[Transaction], Dict[str, Any]]:
        
        buy_threshold = self.kwargs.get('buy_threshold', 0.5)
        sell_threshold = self.kwargs.get('sell_threshold', 0.5)
        
        price_diff = predicted_price - actual_price
        
        decision_details = {
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "diff": price_diff,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold
        }
        
        # Buy signal: predicted price exceeds buy threshold
        if price_diff > buy_threshold and self.balance > 0:
            quantity = self.balance / actual_price
            self.stocks_owned += quantity
            self.balance = 0
            self.last_buy_price = actual_price
            
            transaction = self._create_transaction(
                date=date,
                transaction_type=TransactionType.BUY,
                price=actual_price,
                quantity=quantity,
                reason=f"Strong buy signal: +{price_diff:.2f}€ > {buy_threshold}€ threshold",
                predicted_price=predicted_price
            )
            return TransactionType.BUY, transaction, decision_details
        
        # Sell signal: predicted price below sell threshold
        elif price_diff < -sell_threshold and self.stocks_owned > 0:
            quantity = self.stocks_owned
            self.balance += quantity * actual_price
            self.stocks_owned = 0
            
            profit_loss = ""
            if self.last_buy_price:
                profit = ((actual_price - self.last_buy_price) / self.last_buy_price) * 100
                profit_loss = f" (Profit: {profit:.2f}%)"
            
            transaction = self._create_transaction(
                date=date,
                transaction_type=TransactionType.SELL,
                price=actual_price,
                quantity=quantity,
                reason=f"Strong sell signal: {price_diff:.2f}€ < -{sell_threshold}€ threshold{profit_loss}",
                predicted_price=predicted_price
            )
            return TransactionType.SELL, transaction, decision_details
        
        return TransactionType.HOLD, None, decision_details


class PercentageStrategy(TradingStrategyExecutor):
    """Percentage strategy: trade based on percentage change predictions"""
    
    def execute_trade(
        self, 
        date: str,
        actual_price: float, 
        predicted_price: float,
        predicted_last_day: float
    ) -> Tuple[TransactionType, Optional[Transaction], Dict[str, Any]]:
        
        buy_threshold = self.kwargs.get('buy_threshold', 1.0)  # 1% by default
        sell_threshold = self.kwargs.get('sell_threshold', 1.0)
        
        predicted_change_pct = ((predicted_price - actual_price) / actual_price) * 100
        
        decision_details = {
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "predicted_change_pct": predicted_change_pct,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold
        }
        
        # Buy signal
        if predicted_change_pct > buy_threshold and self.balance > 0:
            quantity = self.balance / actual_price
            self.stocks_owned += quantity
            self.balance = 0
            self.last_buy_price = actual_price
            
            transaction = self._create_transaction(
                date=date,
                transaction_type=TransactionType.BUY,
                price=actual_price,
                quantity=quantity,
                reason=f"Predicted increase: +{predicted_change_pct:.2f}% > {buy_threshold}% threshold",
                predicted_price=predicted_price
            )
            return TransactionType.BUY, transaction, decision_details
        
        # Sell signal
        elif predicted_change_pct < -sell_threshold and self.stocks_owned > 0:
            quantity = self.stocks_owned
            self.balance += quantity * actual_price
            self.stocks_owned = 0
            
            profit_loss = ""
            if self.last_buy_price:
                profit = ((actual_price - self.last_buy_price) / self.last_buy_price) * 100
                profit_loss = f" (Profit: {profit:.2f}%)"
            
            transaction = self._create_transaction(
                date=date,
                transaction_type=TransactionType.SELL,
                price=actual_price,
                quantity=quantity,
                reason=f"Predicted decrease: {predicted_change_pct:.2f}% < -{sell_threshold}% threshold{profit_loss}",
                predicted_price=predicted_price
            )
            return TransactionType.SELL, transaction, decision_details
        
        return TransactionType.HOLD, None, decision_details


class ConservativeStrategy(TradingStrategyExecutor):
    """Conservative strategy: only trade with high confidence and profit protection"""
    
    def execute_trade(
        self, 
        date: str,
        actual_price: float, 
        predicted_price: float,
        predicted_last_day: float
    ) -> Tuple[TransactionType, Optional[Transaction], Dict[str, Any]]:
        
        min_profit_pct = self.kwargs.get('min_profit_percentage', 5.0)  # 5% minimum profit
        buy_threshold = self.kwargs.get('buy_threshold', 2.0)  # 2% predicted increase
        
        predicted_change_pct = ((predicted_price - actual_price) / actual_price) * 100
        
        current_profit_pct = 0.0
        if self.last_buy_price:
            current_profit_pct = ((actual_price - self.last_buy_price) / self.last_buy_price) * 100
            
        decision_details = {
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "predicted_change_pct": predicted_change_pct,
            "buy_threshold": buy_threshold,
            "min_profit_pct": min_profit_pct,
            "current_profit_pct": current_profit_pct
        }
        
        # Buy signal: strong predicted increase
        if predicted_change_pct > buy_threshold and self.balance > 0:
            quantity = self.balance / actual_price
            self.stocks_owned += quantity
            self.balance = 0
            self.last_buy_price = actual_price
            
            transaction = self._create_transaction(
                date=date,
                transaction_type=TransactionType.BUY,
                price=actual_price,
                quantity=quantity,
                reason=f"Conservative buy: +{predicted_change_pct:.2f}% predicted > {buy_threshold}% threshold",
                predicted_price=predicted_price
            )
            return TransactionType.BUY, transaction, decision_details
        
        # Sell signal: either profit target reached or predicted decrease
        elif self.stocks_owned > 0:
            # Sell if profit target reached
            if current_profit_pct >= min_profit_pct:
                quantity = self.stocks_owned
                self.balance += quantity * actual_price
                self.stocks_owned = 0
                
                transaction = self._create_transaction(
                    date=date,
                    transaction_type=TransactionType.SELL,
                    price=actual_price,
                    quantity=quantity,
                    reason=f"Profit target reached: {current_profit_pct:.2f}% >= {min_profit_pct}% target",
                    predicted_price=predicted_price
                )
                return TransactionType.SELL, transaction, decision_details
            
            # Sell if strong predicted decrease
            elif predicted_change_pct < -1.0:
                quantity = self.stocks_owned
                self.balance += quantity * actual_price
                self.stocks_owned = 0
                
                transaction = self._create_transaction(
                    date=date,
                    transaction_type=TransactionType.SELL,
                    price=actual_price,
                    quantity=quantity,
                    reason=f"Predicted decrease: {predicted_change_pct:.2f}% (profit: {current_profit_pct:.2f}%)",
                    predicted_price=predicted_price
                )
                return TransactionType.SELL, transaction, decision_details
        
        return TransactionType.HOLD, None, decision_details


class AggressiveStrategy(TradingStrategyExecutor):
    """Aggressive strategy: trade on small signals with stop-loss"""
    
    def execute_trade(
        self, 
        date: str,
        actual_price: float, 
        predicted_price: float,
        predicted_last_day: float
    ) -> Tuple[TransactionType, Optional[Transaction], Dict[str, Any]]:
        
        buy_threshold = self.kwargs.get('buy_threshold', 0.1)  # 0.1% predicted increase
        max_loss_pct = self.kwargs.get('max_loss_percentage', 5.0)  # 5% max loss
        
        predicted_change_pct = ((predicted_price - actual_price) / actual_price) * 100
        
        current_loss_pct = 0.0
        if self.last_buy_price:
            current_loss_pct = ((actual_price - self.last_buy_price) / self.last_buy_price) * 100
            
        decision_details = {
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "predicted_change_pct": predicted_change_pct,
            "buy_threshold": buy_threshold,
            "max_loss_pct": max_loss_pct,
            "current_loss_pct": current_loss_pct
        }
        
        # Buy signal: any predicted increase
        if predicted_change_pct > buy_threshold and self.balance > 0:
            quantity = self.balance / actual_price
            self.stocks_owned += quantity
            self.balance = 0
            self.last_buy_price = actual_price
            
            transaction = self._create_transaction(
                date=date,
                transaction_type=TransactionType.BUY,
                price=actual_price,
                quantity=quantity,
                reason=f"Aggressive buy: +{predicted_change_pct:.2f}% predicted > {buy_threshold}%",
                predicted_price=predicted_price
            )
            return TransactionType.BUY, transaction, decision_details
        
        # Sell signal: predicted decrease or stop-loss
        elif self.stocks_owned > 0:
            # Stop-loss triggered
            if current_loss_pct <= -max_loss_pct:
                quantity = self.stocks_owned
                self.balance += quantity * actual_price
                self.stocks_owned = 0
                
                transaction = self._create_transaction(
                    date=date,
                    transaction_type=TransactionType.SELL,
                    price=actual_price,
                    quantity=quantity,
                    reason=f"STOP-LOSS: {current_loss_pct:.2f}% loss (limit: -{max_loss_pct}%)",
                    predicted_price=predicted_price
                )
                return TransactionType.SELL, transaction, decision_details
            
            # Sell on predicted decrease
            elif predicted_change_pct < 0:
                quantity = self.stocks_owned
                self.balance += quantity * actual_price
                self.stocks_owned = 0
                
                transaction = self._create_transaction(
                    date=date,
                    transaction_type=TransactionType.SELL,
                    price=actual_price,
                    quantity=quantity,
                    reason=f"Predicted decrease: {predicted_change_pct:.2f}%",
                    predicted_price=predicted_price
                )
                return TransactionType.SELL, transaction, decision_details
        
        return TransactionType.HOLD, None, decision_details


def get_strategy_executor(strategy: TradingStrategy, initial_balance: float, **kwargs) -> TradingStrategyExecutor:
    """Factory function to get the appropriate strategy executor"""
    strategies = {
        TradingStrategy.SIMPLE: SimpleStrategy,
        TradingStrategy.THRESHOLD: ThresholdStrategy,
        TradingStrategy.PERCENTAGE: PercentageStrategy,
        TradingStrategy.CONSERVATIVE: ConservativeStrategy,
        TradingStrategy.AGGRESSIVE: AggressiveStrategy,
    }
    
    strategy_class = strategies.get(strategy, SimpleStrategy)
    return strategy_class(initial_balance, **kwargs)
