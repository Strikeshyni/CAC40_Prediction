import json
import os
from models.model_lstm_v2 import model_lstm_v2
from models.model_transformer import model_transformer
from models.model_xgboost import model_xgboost

def tune_and_save():
    stock_names = ["AIR.PA", "ENGI.PA", "SAN.PA", "BNP.PA"]
    from_date = "2015-01-01"
    to_date = "2025-11-28"
    
    hyperparameters = {}
    
    if os.path.exists("hyperparameters/best_hyperparameters.json"):
        with open("hyperparameters/best_hyperparameters.json", "r") as f:
            hyperparameters = json.load(f)
    
    for stock_name in stock_names:
        if stock_name not in hyperparameters:
            hyperparameters[stock_name] = {}

        print(f"Tuning LSTM v2 for {stock_name}...")
        lstm = model_lstm_v2(
            model_name="tuning_lstm_v2",
            stock_name=stock_name,
            from_date=from_date,
            to_date=to_date,
            global_tuning=True,
            epochs=30,
            verbose=True
        )
        hyperparameters[stock_name]["model_lstm_v2"] = lstm.best_hyperparameters
        print("LSTM v2 tuning complete.")

        # print(f"Tuning Transformer for {stock_name}...")
        # transformer = model_transformer(
        #     model_name="tuning_transformer",
        #     stock_name=stock_name,
        #     from_date=from_date,
        #     to_date=to_date,
        #     global_tuning=True,
        #     epochs=30,
        #     verbose=True
        # )
        # hyperparameters[stock_name]["model_transformer"] = transformer.best_hyperparameters
        # print("Transformer tuning complete.")

        # print(f"Tuning XGBoost for {stock_name}...")
        # xgboost = model_xgboost(
        #     model_name="tuning_xgboost",
        #     stock_name=stock_name,
        #     from_date=from_date,
        #     to_date=to_date,
        #     global_tuning=True,
        #     n_iter=30
        # )
        # hyperparameters[stock_name]["model_xgboost"] = xgboost.best_params
        # print("XGBoost tuning complete.")

    with open("hyperparameters/best_hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=4)
    
    print("All hyperparameters saved to hyperparameters/best_hyperparameters.json")

if __name__ == "__main__":
    tune_and_save()
