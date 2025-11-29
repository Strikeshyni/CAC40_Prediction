from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Conv1D
from keras.callbacks import EarlyStopping
from web_scrapper.scrapper import get_closing_prices
from models.callbacks import CustomVerboseCallback

class model_transformer:
    def __init__(self, model_name, stock_name, from_date, to_date, train_size_percent=0.6, val_size_percent=0.2, time_step_train_split=100, global_tuning=False, epochs=50, verbose=True, progress_callback=None, hyperparameters=None):
        self.model_name = model_name
        self.stock_name = stock_name
        self.global_tuning = global_tuning
        self.epochs = epochs
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.hyperparameters = hyperparameters
        
        print(f"Loading data for {stock_name} from {from_date} to {to_date}...")
        try:
            self.stock_data = get_closing_prices(from_date, to_date, stock_name)
        except ValueError as e:
            print(e)
            raise ValueError("No data on this day, CAC 40 was closed")
        print(f"Data loaded successfully. {len(self.stock_data)} records found.")
        
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

    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def _build_model(self, hp):
        inputs = Input(shape=(self.time_step, 1))
        
        head_size = hp.Int('head_size', min_value=32, max_value=128, step=32)
        num_heads = hp.Int('num_heads', min_value=2, max_value=4, step=1)
        ff_dim = hp.Int('ff_dim', min_value=32, max_value=128, step=32)
        num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=4, step=1)
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.3, step=0.1)
        
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self._transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout)(x)
        x = Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation="relu")(x)
        x = Dropout(dropout)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def tune_model(self):
        if self.hyperparameters:
            print("Using provided hyperparameters, skipping tuning.")
            hp = kt.HyperParameters()
            hp.values = self.hyperparameters
            self.best_model = self._build_model(hp)
            self.best_hyperparameters = self.hyperparameters
            return

        print("Starting hyperparameter tuning...")
        # Use Keras Tuner to find the best hyperparameters
        with tf.device('/GPU:0'):
            tuner = kt.Hyperband(
                self._build_model,
                objective='val_loss',
                max_epochs=self.epochs,
                factor=3,
                directory="global_tuner" if self.global_tuning else f'{self.model_name}/tuner_results',
                project_name=f'stock_prediction_transformer_{self.stock_name}'
            )

        # Perform the search
        tuner_verbose = 1 if self.verbose else 0
        
        # For tuning, we allocate 0% to 50% of model progress
        if self.progress_callback:
            self.progress_callback(0.0)
            
        tuner.search(self.X_train, self.y_train, epochs=self.epochs, validation_data=(self.X_val, self.y_val), verbose=tuner_verbose)
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.best_hyperparameters = best_hps.values
        self.best_model = tuner.hypermodel.build(best_hps)
        
        print("Tuning completed.")
        if self.verbose:
            print("Best Hyperparameters:", best_hps.values)
            print(self.best_model.summary())
    
    def train_model(self):
        print("Starting model training...")
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        
        if not self.verbose or self.progress_callback:
            callbacks.append(CustomVerboseCallback(
                total_epochs=100, 
                print_every=10, 
                progress_callback=self.progress_callback,
                start_progress=0.5,
                end_progress=1.0
            ))
            
        verbose_fit = 1 if self.verbose else 0
        self.best_model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=100, batch_size=32, callbacks=callbacks, verbose=verbose_fit)
        print("Model training completed.")

    def predict(self, data):
        # data shape should be (samples, time_step)
        # reshape for Transformer (samples, time_step, 1)
        data = data.reshape((data.shape[0], data.shape[1], 1))
        prediction = self.best_model.predict(data)
        return self.scaler.inverse_transform(prediction)
