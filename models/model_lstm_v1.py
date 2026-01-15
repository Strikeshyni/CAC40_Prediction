from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from web_scrapper.scrapper import get_closing_prices

class model_v1:
    def __init__(self, model_name, stock_name, from_date, to_date, train_size_percent=0.6, val_size_percent=0.2, time_step_train_split=100, global_tuning=False, epochs=50, max_trials=10):
        self.model_name = model_name
        self.stock_name = stock_name
        self.global_tuning = global_tuning
        self.epochs = epochs
        self.max_trials = max_trials
        try:
            self.stock_data = get_closing_prices(from_date, to_date, stock_name)
        except ValueError as e:
            print(e)
            raise ValueError("No data on this day, CAC 40 was closed")
        self.scaled_data = None
        self.scaler = None
        self.scale_data(self.stock_data)
        self.best_model = None
        self.time_step = time_step_train_split
        self.X, self.y = self.create_dataset(self.scaled_data, self.time_step)
        self.train_size = int(len(self.X) * train_size_percent)  # 60% for training
        self.val_size = int(len(self.X) * val_size_percent)    # 20% for validation and keep 20% for testing
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.split_data()
        # print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        # print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
        # print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
        self.tune_model()
        self.train_model()
        #self.test_model()

    def scale_data(self, stock_data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(stock_data.values.reshape(-1, 1))
    
    def create_dataset(self, dataset, time_step=300):
        X, Y = [], []
        for i in range(len(dataset)-time_step):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        # print(self.scaler.inverse_transform(a.reshape(-1, 1)))
        # print(self.scaler.inverse_transform(dataset[i + time_step, 0].reshape(-1, 1)))
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
        def build_model(hp):
            model = Sequential()
            # Tune the number of LSTM units
            lstm_units = hp.Int('lstm_units', min_value=128, max_value=256, step=32)
            dense_units = hp.Int('dense_units', min_value=64, max_value=128, step=32)
            activation_function = hp.Choice('activation_function', values=['tanh']) # Remove the 'relu' option because never used

            model.add(LSTM(units=lstm_units, activation=activation_function, return_sequences=True, input_shape=(self.time_step, 1)))
            model.add(LSTM(units=lstm_units, activation=activation_function))
            model.add(Dense(units=dense_units, activation='relu'))
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model

        # Use Keras Tuner to find the best hyperparameters
        with tf.device('/GPU:0'):
            tuner = kt.Hyperband(
                build_model,
                objective='val_loss',
                max_epochs=self.epochs,
                factor=3,
                directory="global_tuner" if self.global_tuning else f'{self.model_name}/tuner_results',
                project_name=f'stock_prediction_{self.stock_name}'
            )

        # Perform the search
        tuner.search(self.X_train, self.y_train, epochs=self.epochs, validation_data=(self.X_val, self.y_val))

        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best Hyperparameters:", best_hps.values)

        # Build the model with the best hyperparameters
        self.best_model = tuner.hypermodel.build(best_hps)
        print(self.best_model.summary())
    
    def train_model(self):
        # Build
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Surveille la perte sur l'ensemble de validation
            patience=10,         # Arrête après 10 epochs sans amélioration
            restore_best_weights=True  # Restaure les poids du meilleur modèle
        )

        with tf.device('/GPU:0'):
            history = self.best_model.fit(
                self.X_train, self.y_train,
                epochs=100,  # Nombre maximum d'epochs
                batch_size=32,
                validation_data=(self.X_val, self.y_val),
                callbacks=[early_stopping],
                verbose=1
            )
    
    def test_model(self):
        print(self.best_model.evaluate(self.X_test, self.y_test))

    def predict(self, data):
        predictions = self.best_model.predict(data)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions