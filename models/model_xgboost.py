from sklearn.preprocessing import MinMaxScaler
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from web_scrapper.scrapper import get_closing_prices
import joblib
import os

class model_xgboost:
    def __init__(self, model_name, stock_name, from_date, to_date, train_size_percent=0.6, val_size_percent=0.2, time_step_train_split=100, global_tuning=False, n_iter=10):
        self.model_name = model_name
        self.stock_name = stock_name
        self.global_tuning = global_tuning
        self.n_iter = n_iter
        try:
            self.stock_data = get_closing_prices(from_date, to_date, stock_name)
        except ValueError as e:
            print(e)
            raise ValueError("Loaded data is same as previous day")
        self.scaled_data = None
        self.scaler = None
        self.scale_data(self.stock_data)
        self.best_model = None
        self.time_step = time_step_train_split
        self.X, self.y = self.create_dataset(self.scaled_data, self.time_step)
        self.train_size = int(len(self.X) * train_size_percent)
        self.val_size = int(len(self.X) * val_size_percent)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.split_data()
        
        self.tune_model()
        self.train_model()

    def scale_data(self, stock_data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(stock_data.values.reshape(-1, 1))
    
    def create_dataset(self, dataset, time_step=300):
        X, Y = [], []
        for i in range(len(dataset)-time_step):
            a = dataset[i:(i+time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)
    
    def split_data(self):
        X_train = self.X[:self.train_size]
        y_train = self.y[:self.train_size]

        X_val = self.X[self.train_size:self.train_size + self.val_size]
        y_val = self.y[self.train_size:self.train_size + self.val_size]

        X_test = self.X[self.train_size + self.val_size:]
        y_test = self.y[self.train_size + self.val_size:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def tune_model(self):
        # XGBoost Hyperparameter tuning using RandomizedSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda' if self.check_gpu() else 'cpu')
        
        # Combine train and val for CV or just use train
        # Using RandomizedSearchCV
        random_search = RandomizedSearchCV(
            xgb_model, 
            param_distributions=param_grid, 
            n_iter=self.n_iter, 
            scoring='neg_mean_squared_error', 
            cv=3, 
            verbose=1, 
            n_jobs=-1
        )
        
        print("Tuning XGBoost model...")
        random_search.fit(self.X_train, self.y_train)
        
        self.best_params = random_search.best_params_
        print("Best Hyperparameters:", self.best_params)
        
        self.best_model = random_search.best_estimator_

    def check_gpu(self):
        try:
            import subprocess
            subprocess.check_output('nvidia-smi')
            return True
        except Exception:
            return False

    def train_model(self):
        # In XGBoost with sklearn API, fit is already training. 
        # But we can retrain on train+val if we want, or just keep the best estimator from CV.
        # Here we will retrain on train set with early stopping using validation set
        
        self.best_model.set_params(**self.best_params)
        self.best_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        print("XGBoost training complete.")

    def predict(self, data):
        # data shape should be (samples, time_step)
        # XGBoost expects 2D array
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], data.shape[1])
            
        prediction = self.best_model.predict(data)
        # XGBoost predict returns 1D array if n_targets=1
        prediction = prediction.reshape(-1, 1)
        return self.scaler.inverse_transform(prediction)
