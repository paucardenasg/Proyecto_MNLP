import optuna
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Flatten, Conv1D, LSTM, Input, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


class NN_maker:
    
    def __init__(self, data, n_steps, horizont) -> None:
        self.data = data
        self.n_steps = n_steps
        self.plot_horizon = horizont

    def transform_data(self):
        data = self.data
        col = data.columns[0]
        data[col] = np.log(data[col])
        self.data = data
        return("Se transformó a un logaritmo")

    def plot_serie(self):
        data = self.data
        col = data.columns[0]
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        sns.boxplot(x=data[col], ax=ax[0])
        sns.histplot(x=data[col], ax=ax[1])
        sns.lineplot(data=data, ax=ax[2])
        ax[0].set(title="Boxplot", xlabel=None)
        ax[1].set(title="Histograma", xlabel=None)
        ax[2].set(title="Diagrama de línea", xlabel=None)
        plt.xticks(rotation=90)
        plt.tight_layout()
    
    @staticmethod
    def series_to_supervised(data, n_in, n_out=1, conv=False):
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))

        # put it all together
        agg = pd.concat(cols, axis=1)

        # drop rows with NaN values
        agg.dropna(inplace=True)

        agg = agg.values

        X, y = agg[:, :-1], agg[:, -1]
        # Convertir valores a forma tensorial
        if conv:
            X = X.reshape((X.shape[0], X.shape[1], 1))
            y = y.reshape((y.shape[0], 1))
        
        return X, y
    

    def train_val_test_split(self, train_ratio=0.8, val_ratio=0.10, conv=False, norm=False):
        # Usamos datos y n_steps global
        data = self.data
        n_steps = self.n_steps

        # Definimos los tamaños.
        m = len(data)
        train_size = int(train_ratio * m)
        val_size = int(val_ratio * m)
        
        # Dividir los datos en conjuntos de entrenamiento, validación y prueba
        train = data[:train_size]
        val = data[train_size:train_size+val_size]
        test = data[train_size+val_size:]

        if norm:
            scaler = StandardScaler()
            train = scaler.fit_transform(train.values.reshape(-1, 1))
            val = scaler.transform(val.values.reshape(-1, 1))
            test = scaler.transform(test.values.reshape(-1, 1))

            # Convertimos en supervisado y dividimos en "x" y "y"
            X_train, y_train = NN_maker.series_to_supervised(train, n_steps, conv=conv)
            X_val, y_val = NN_maker.series_to_supervised(val, n_steps, conv=conv)
            X_test, y_test = NN_maker.series_to_supervised(test, n_steps, conv=conv)
            return X_train, X_val, X_test, y_train, y_val, y_test, scaler
        
        # Convertimos en supervisado y dividimos en "x" y "y"
        X_train, y_train = NN_maker.series_to_supervised(train, n_steps, conv=conv)
        X_val, y_val = NN_maker.series_to_supervised(val, n_steps, conv=conv)
        X_test, y_test = NN_maker.series_to_supervised(test, n_steps, conv=conv)
        return X_train, X_val, X_test, y_train, y_val, y_test
    

    def train_models(self, model, X_train, X_val, X_test, y_train, y_val, y_test, scaler=None, plot=False, norm=False, log=False):
        
        # Definimos Earlystopping
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15)
        
        # Compilamos el modelo
        model.compile( optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        
        # Entrenamos
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=500, verbose=False,
                            callbacks=earlystop)
        
        # Generar predicciones
        y_pred = model.predict(X_test, verbose=False)
        # Calculamos métricas
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print("R2:", r2)
        print("MAE:", mae)

        # Crear dataframe con los datos de test y las predicciones
        df_pred = pd.DataFrame({'test': y_test.flatten(), 
                                'pred': y_pred.flatten()}, 
                                index=self.data.index[-len(y_test):])
        if norm:
            df_pred[['test', 'pred']] = scaler.inverse_transform(df_pred[['test', 'pred']])

        if log:
            df_pred[['test', 'pred']] = np.exp(df_pred[["test", 'pred']])

        if plot:
            fig, ax = plt.subplots(3, 1, figsize=(8, 8))
            pd.DataFrame(history.history).plot(ax=ax[0])
            df_pred.plot(ax=ax[1])
            df_pred['error'] = df_pred['test'] - df_pred['pred']
            df_pred["error"].plot(ax=ax[2])
            ax[0].set(title="Loss vs Val loss", xlabel=None)
            ax[1].set(title="Prediccion", xlabel=None)
            ax[2].set(title="Error", xlabel=None)
            plt.tight_layout()
        score = model.evaluate(X_test, y_test, verbose=False)
        return model


    def MLP_builder(self, num_hidden_layers, X_train, X_val, X_test, 
                    y_train, y_val, y_test, num_neurons, scaler=None, 
                    dropout_rate=None, plot=False, norm=False, log=False):
        
        model = Sequential()
        # Agregar la primera capa oculta
        model.add(Dense(num_neurons, input_dim=X_train.shape[1], activation='relu'))

        # Se agrega capa dropout si se añade un valor como input
        if dropout_rate:
            model.add(Dropout(dropout_rate))

        # Agregar capas ocultas adicionales
        for _ in range(num_hidden_layers-1):
            model.add(Dense(num_neurons, activation='relu'))
        
        # Se agrega capa dropout si se añade un valor como input
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        # Agregar la capa de salida
        model.add(Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse')
        
        model_fit = NN_maker.train_models(self=self, model=model, X_train=X_train,  X_val=X_val, X_test=X_test, 
                                      y_train=y_train, y_val=y_val, y_test=y_test, scaler=scaler, plot=plot, norm=norm, log=log)
        return model_fit
    
    def cnn_builder(self, num_hidden_layers, num_neurons, num_filters, kernel_size,
                    pool_size, X_train, X_val, X_test, y_train, y_val, y_test, scaler=None, 
                    dropout_rate=None, plot=False, norm=False, log=False):
        
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        # Agregar la capa de convolución
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
        # Agregar la capa de pooling
        model.add(MaxPooling1D(pool_size=pool_size))
        
        # Agregar capas de convolución y pooling adicionales
        for _ in range(num_hidden_layers-1):
            model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=pool_size))
        
        # Agregar la capa de aplanamiento
        model.add(Flatten())
        
        # Agregar capas ocultas totalmente conectadas
        for _ in range(num_hidden_layers):
            model.add(Dense(num_neurons, activation='relu'))
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        
        # Agregar la capa de salida
        model.add(Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse')
        
        model_fit = NN_maker.train_models(self=self, model=model, X_train=X_train,  X_val=X_val, X_test=X_test, 
                                      y_train=y_train, y_val=y_val, y_test=y_test, scaler=scaler, plot=plot, norm=norm, log=log)
        return model_fit


    def lstm_builder(self, units_lstm, capas_ocultas_lstm, capas_ocultas_dense, units_dense, 
                     X_train, X_val, X_test, y_train, y_val, y_test, scaler=None, plot=False, dropout=None, norm=False, log=False):
        # Definir el modelo de red neuronal
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        # Capas LSTM
        for _ in range(capas_ocultas_lstm):
            model.add(LSTM(units_lstm))
        # Capas ocultas
        for _ in range(capas_ocultas_dense):
            model.add(Dense(units_dense, activation='relu'))
        # Capa Dropout
        if dropout:    
            model.add(Dropout(dropout))
        # Capa de salida
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse')
        
        model_fit = NN_maker.train_models(self=self, model=model, X_train=X_train,  X_val=X_val, X_test=X_test, 
                                      y_train=y_train, y_val=y_val, y_test=y_test, scaler=scaler, plot=plot, norm=norm, log=log)
        return model_fit

    def cnn_lstm_builder(self, blocks, filters, units_lstm,  X_train, X_val, X_test, y_train, y_val, y_test, 
                         scaler=None, plot=False, dropout=None, norm=False, log=False):
        
        model = Sequential()
        model.add(Input(shape=(None, X_train.shape[2], X_train.shape[3])))
        for _ in range(blocks):
            model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=3, activation='relu')))
            model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(units=units_lstm, return_sequences=False))
        if dropout:
            model.add(Dropout(dropout))
        
        model.add(Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse')
        
        model_fit = NN_maker.train_models(self=self, model=model, X_train=X_train,  X_val=X_val, X_test=X_test, 
                                      y_train=y_train, y_val=y_val, y_test=y_test, scaler=scaler, plot=plot, norm=norm, log=log)
        return model_fit
    
    @staticmethod
    def objective_conv_lstm(trial,  X_train, X_val, X_test, y_train, y_val, y_test):
        # define hyperparameters to optimize
        n_filters = trial.suggest_int('n_filters', 32, 128)
        n_lstm_units = trial.suggest_int('n_lstm_units', 32, 128)
        n_epochs = trial.suggest_int('n_epochs', 100, 1000, step=100)

        # Creamos el modelo
        model = Sequential()
        model.add(Input(shape=(None, X_train.shape[2], X_train.shape[3])))
        model.add(TimeDistributed(Conv1D(n_filters, 1, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(n_lstm_units, activation='relu'))
        model.add(Dense(1))

        # Compilamos el modelo
        model.compile( optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        
        # Entrenamos
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=n_epochs, verbose=False)
        # fit 
        val_loss = model.evaluate(X_test, y_test, verbose=0)

        return val_loss 
    
    @staticmethod
    def objective_mlp(trial, X_train, X_val, X_test, y_train, y_val, y_test):
        # Definir los rangos de búsqueda de los hiperparámetros
        num_neurons = trial.suggest_int('num_neurons', 32, 512)
        num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 4)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0, 0.5)
        n_epochs = trial.suggest_int('epochs', 10, 100)

        model = Sequential()
        # Agregar la primera capa oculta
        model.add(Dense(num_neurons, input_dim=X_train.shape[1], activation='relu'))
        # Agregar capas ocultas adicionales
        for _ in range(num_hidden_layers-1):
            model.add(Dense(num_neurons, activation='relu'))
        # Se agrega capa dropout si se añade un valor como input
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        # Agregar la capa de salida
        model.add(Dense(1))
        # Compilamos el modelo
        model.compile( optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        # Entrenamos
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=n_epochs, verbose=False)
        # fit 
        val_loss = model.evaluate(X_test, y_test, verbose=0)
        return val_loss
    
    @staticmethod
    def create_sequences(array, n_steps, conv=False):
        X = []
        for i in range(len(array)-n_steps):
            X.append(array[i:i+n_steps])
        X = np.array(X)
        if conv:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        return X
    
    def plot_horizon_log(self, model, conv=False):
        horizont = int(self.horizont)
        n_features = 1
        n_seq = 4
        n_steps = 6
        y_preds = []
        datos = self.data
        datos = datos["Serie"].values
        for _ in range(horizont):
            X_Trans = np.log(datos)
            X = NN_maker.create_sequences(X_Trans, self.n_steps, conv=conv)
            X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
            y_pred = model.predict(X, verbose=False)
            y_pred = np.exp(y_pred)
            y_preds.append(y_pred[-1])
            datos = np.append(datos, y_pred[-1])
        
        y_true = self.data['Serie'].values
        # Graficamos la serie de tiempo original junto con las predicciones
        plt.plot(y_true, label='Serie de tiempo')
        plt.plot(np.arange(len(y_true), len(y_true)+horizont), y_preds, label=f'Prediccion en horizonte: {horizont}')
        # Personalizamos el gráfico
        plt.title('Serie de tiempo con predicciones')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        # Mostramos el gráfico
        plt.show()


    def plot_horizon(self, model, scaler=False, conv=False):
        horizont = int(self.horizont)
        n_features = 1
        n_seq = 4
        n_steps = 6
        y_preds = []
        datos = self.data
        col = datos.columns[0]
        datos = datos[col].values
        for _ in range(horizont):
            X_scaled = scaler.transform(datos.reshape(-1, 1))
            X = NN_maker.create_sequences(X_scaled, self.n_steps, conv=conv)
            X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
            y_pred = model.predict(X, verbose=False)
            y_pred = scaler.inverse_transform(y_pred)
            y_preds.append(y_pred[-1])
            datos = np.append(datos, y_pred[-1])
        
        y_true = self.data[col].values
        # Graficamos la serie de tiempo original junto con las predicciones
        plt.plot(y_true, label='Serie de tiempo')
        plt.plot(np.arange(len(y_true), len(y_true)+horizont), y_preds, label=f'Prediccion en horizonte: {horizont}')
        # Personalizamos el gráfico
        plt.title('Serie de tiempo con predicciones')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        # Mostramos el gráfico
        plt.show()
    
    def conv_lstm_optimal(self, n_filters, n_lstm_units, n_epochs, scaler=False, log=False):
        # Creamos el modelo
        X, y = NN_maker.series_to_supervised(self.data, self.n_steps, conv=True)
        n_features = 1
        n_seq = 4
        n_steps = 6
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
        model = Sequential()
        model.add(Input(shape=(None, X.shape[2], X.shape[3])))
        model.add(TimeDistributed(Conv1D(n_filters, 1, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(n_lstm_units, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Compilamos el modelo
        model.compile( optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        # Entrenamos
        history = model.fit(X, y, epochs=n_epochs, verbose=False)
        if log:
            NN_maker.plot_horizon_log(self, model, conv=True)
        else: 
            NN_maker.plot_horizon(self, model, conv=True, scaler=scaler)
    
    def mlp_optimal(self, num_neurons, num_hidden_layers, dropout_rate, n_epochs, scaler):
        X, y = NN_maker.series_to_supervised(self.data, self.n_steps, conv=False)
        model = Sequential()
        # Agregar la primera capa oculta
        model.add(Dense(num_neurons, input_dim=X.shape[1], activation='relu'))
        # Agregar capas ocultas adicionales
        for _ in range(num_hidden_layers-1):
            model.add(Dense(num_neurons, activation='relu'))
        # Se agrega capa dropout si se añade un valor como input
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        # Agregar la capa de salida
        model.add(Dense(1))
        # Compilamos el modelo
        model.compile( optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        # Entrenamos
        history = model.fit(X, y,
                            epochs=n_epochs, verbose=False)
        NN_maker.plot_horizon(self, model, scaler=scaler)



    
    
    