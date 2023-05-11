# Librerías
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Flatten, Conv1D, MaxPooling1D, Input, Bidirectional, TimeDistributed

# -----------------------------------------------------------------------------------------------------------

# Funciones para preprocesamiento de datos

# Dividir una secuencia multivariada en muestras
def split_multivariate_sequence(sequence: np.ndarray, n_steps: int):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Encontrar el final de este patrón
        end_ix = i + n_steps
        # Comprobar si estamos más allá de la secuencia
        if end_ix > len(sequence):
            break
        # Reunir partes de entrada y salida del patrón
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    # Haciendo tensor a y
    y = np.expand_dims(y, axis=1)
    # Mostrando el tamaño
    print(f'Forma de X: {np.array(X).shape}')
    print(f'Forma de y: {np.array(y).shape}')
    # Regresando los valores
    return np.array(X), np.array(y)

# Función para separar en Train y Test
def train_test_split_multi(X: np.array, y: np.array, train: float = 0.8):
    # Calculando el tamaño de train
    train_size = int(len(X) * train)
    # Separando en train y test
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # Obteniendo el index de cada uno
    index_train = np.arange(0, train_size)
    index_test = np.arange(train_size, len(X))
    # Mostrando los tamaños resultantes
    print(f'Entrenamiento:\n- X: {X_train.shape}\n- y: {y_train.shape}')
    print(f'\nPrueba:\n- X: {X_test.shape}\n- y: {y_test.shape}')
    # Regresando los valores
    return X_train, X_test, y_train, y_test

# -----------------------------------------------------------------------------------------------------------

# Funciones para modelos
# Creando una función para crear el modelo MLP
def gen_MLP_model(X, y, val_split, input_shape,
                  activation, num_layers, num_neurons, 
                  optimizer, lr, loss, metrics, 
                  patience, epochs, verbose, 
                  X_test, y_test, index,
                  plot_history=True):
    # Creando el modelo secuencial
    model = Sequential()
    # Agregando la capa de entrada
    model.add(Input(shape=input_shape))
    # Agregando las capas ocultas
    for i in range(num_layers):
        model.add(Dense(num_neurons, activation=activation))
    # Agregando una capa Flatten
    model.add(Flatten())
    # Agregando la capa de salida
    model.add(Dense(1))
    # Compilando el modelo
    model.compile(optimizer=getattr(optimizers, optimizer)(learning_rate=lr),
                  loss=loss, 
                  metrics=metrics)
    # Si no hay paciencia no se pone el callback
    if patience == 0:
        # Entrenando el modelo
        history = model.fit(X, y, 
                            validation_split=val_split,
                            epochs=epochs,
                            verbose=verbose)
    # Si hay paciencia se pone el callback
    else:
        # Entrenando el modelo
        history = model.fit(X, y, 
                            validation_split=val_split,
                            epochs=epochs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
                            verbose=verbose)
    # Obteniendo el último estado de la métrica
    ultimo_estado = np.round(history.history[f"val_{metrics[0]}"][-1], decimals=4)
    # Obteniendo las predicciones
    y_pred = model.predict(X_test)
    # Obteniendo el r2
    r2 = np.round(r2_score(y_test, y_pred), decimals=4)
    # Mostrando lo obtenido
    print(f"El último estado de la métrica {metrics[0]} es: {ultimo_estado}")
    print(f"El R2 del modelo es: {r2}")
    # Crear dataframe con los datos de test y las predicciones
    df_pred = pd.DataFrame({'test': y_test.flatten(), 
                            'pred': y_pred.flatten()}, 
                            index=index)
    # Graficando el historial
    if plot_history:
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        pd.DataFrame(history.history).plot(ax=ax[0])
        df_pred.plot(ax=ax[1])
        df_pred['error'] = df_pred['test'] - df_pred['pred']
        df_pred["error"].plot(ax=ax[2])
        ax[0].set(title="Loss vs Val loss", xlabel=None)
        ax[1].set(title="Prediccion", xlabel=None)
        ax[2].set(title="Error", xlabel=None)
        plt.tight_layout()
    return model, history

# Función para crear una red CNN
def gen_CNN_model(X, y, val_split, input_shape, 
                  num_layers_cnn, num_filters, kernel_size, padding,
                  activation, num_layers_dense, num_neurons, 
                  optimizer, lr, loss, metrics,
                  patience, epochs, verbose,
                  X_test, y_test, index,
                  plot_history):
    # Definición del modelo
    model = Sequential()
    # Agregando la capa de entrada
    model.add(Input(shape=input_shape))
    # Creando un ciclo para agregar las capas CNN
    for i in range(num_layers_cnn):
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation=activation, padding=padding))
        model.add(MaxPooling1D(pool_size=2, padding=padding))
    # Aplanando la salida
    model.add(Flatten())
    # Creando un ciclo para agregar las capas Dense
    for i in range(num_layers_dense):
        model.add(Dense(num_neurons, activation=activation))
    # Agregando la capa de salida
    model.add(Dense(1))
    # Compilando el modelo
    model.compile(optimizer=getattr(optimizers, optimizer)(learning_rate=lr),
                  loss=loss, 
                  metrics=metrics)
    # Si no hay paciencia no se pone el callback
    if patience == 0:
        # Entrenando el modelo
        history = model.fit(X, y, 
                            validation_split=val_split,
                            epochs=epochs,
                            verbose=verbose)
    # Si hay paciencia se pone el callback
    else:
        # Entrenando el modelo
        history = model.fit(X, y, 
                            validation_split=val_split,
                            epochs=epochs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
                            verbose=verbose)
    # Obteniendo el último estado de la métrica
    ultimo_estado = np.round(history.history[f"val_{metrics[0]}"][-1], decimals=4)
    # Obteniendo las predicciones
    y_pred = model.predict(X_test)
    # Obteniendo el R2
    r2 = np.round(r2_score(y_test, y_pred), decimals=4)
    # Mostrando lo obtenido
    print(f"El último estado de la métrica {metrics[0]} es: {ultimo_estado}")
    print(f"El R2 del modelo es: {r2}")
    # Crear dataframe con los datos de test y las predicciones
    df_pred = pd.DataFrame({'test': y_test.flatten(), 
                            'pred': y_pred.flatten()}, 
                            index=index)
    # Graficando el historial
    if plot_history:
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        pd.DataFrame(history.history).plot(ax=ax[0])
        df_pred.plot(ax=ax[1])
        df_pred['error'] = df_pred['test'] - df_pred['pred']
        df_pred["error"].plot(ax=ax[2])
        ax[0].set(title="Loss vs Val loss", xlabel=None)
        ax[1].set(title="Prediccion", xlabel=None)
        ax[2].set(title="Error", xlabel=None)
        plt.tight_layout()
    return model, history

# Función para generar el modelo LSTM
def gen_LSTM_model(X, y, val_split, input_shape,
                  num_layers_lstm, activation_lstm, num_units_lstm, bidireccional,
                  activation, num_layers_dense, num_neurons, 
                  optimizer, lr, loss, metrics, 
                  patience, epochs, verbose,
                  X_test, y_test, index,
                  plot_history):
    # Definición del modelo
    model = Sequential()
    # Agregando capa de entrada
    model.add(Input(shape=input_shape))
    # Agregando las capas LSTM
    for i in range(num_layers_lstm):
        if bidireccional:
            model.add(Bidirectional(LSTM(units=num_units_lstm, activation=activation_lstm, return_sequences=True)))
        else:
            model.add(LSTM(units=num_units_lstm, activation=activation_lstm, return_sequences=True))
    # Flatten
    model.add(Flatten())
    # Agregando las capas Dense
    for i in range(num_layers_dense):
        model.add(Dense(num_neurons, activation=activation))
    
    # Agregando capa de salida
    model.add(Dense(1))
    # Compilando el modelo
    model.compile(optimizer=getattr(optimizers, optimizer)(learning_rate=lr),
                  loss=loss, 
                  metrics=metrics)
    # Si no hay paciencia no se pone el callback
    if patience == 0:
        # Entrenando el modelo
        history = model.fit(X, y, 
                            validation_split=val_split,
                            epochs=epochs,
                            verbose=verbose)
    # Si hay paciencia se pone el callback
    else:
        # Entrenando el modelo
        history = model.fit(X, y, 
                            validation_split=val_split,
                            epochs=epochs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
                            verbose=verbose)
    # Obteniendo el último estado de la métrica
    ultimo_estado = np.round(history.history[f"val_{metrics[0]}"][-1], decimals=4)
    # Obteniendo las predicciones
    y_pred = model.predict(X_test)
    # Obteniendo el R2
    r2 = np.round(r2_score(y_test, y_pred), decimals=4)
    # Mostrando lo obtenido
    print(f"El último estado de la métrica {metrics[0]} es: {ultimo_estado}")
    print(f"El R2 del modelo es: {r2}")
    # Crear dataframe con los datos de test y las predicciones
    df_pred = pd.DataFrame({'test': y_test.flatten(), 
                            'pred': y_pred.flatten()}, 
                            index=index)
    # Graficando el historial
    if plot_history:
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        pd.DataFrame(history.history).plot(ax=ax[0])
        df_pred.plot(ax=ax[1])
        df_pred['error'] = df_pred['test'] - df_pred['pred']
        df_pred["error"].plot(ax=ax[2])
        ax[0].set(title="Loss vs Val loss", xlabel=None)
        ax[1].set(title="Prediccion", xlabel=None)
        ax[2].set(title="Error", xlabel=None)
        plt.tight_layout()
    return model, history


# Función para crear una arquitectura de red neuronal CNN - LSTM
def gen_CNN_LSTM_model(X, y, val_split, input_shape,
                       num_layers_cnn, num_filters, kernel_size, padding,
                       num_layers_lstm, activation_lstm, num_units_lstm,
                       activation, num_layers_dense, num_neurons, 
                       optimizer, lr, loss, metrics, 
                       patience, epochs, verbose,
                       X_test, y_test, index,
                       plot_history):
    # Reshape de los datos para poder usar la arquitectura
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], 1)

    # Creando el modelo
    model = Sequential()
    # Ciclo para agregar las capas CNN
    for i in range(num_layers_cnn):
        model.add(TimeDistributed(Conv1D(filters=num_filters, kernel_size=kernel_size, padding=padding, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2, padding=padding)))
    # Aplanando la salida de las capas CNN
    model.add(TimeDistributed(Flatten()))
    # Agregando las capas LSTM
    for i in range(num_layers_lstm):
        model.add(LSTM(num_units_lstm, activation=activation_lstm, return_sequences=True))
    # Capa de salida
    model.add(LSTM(num_units_lstm, activation=activation_lstm))
    model.add(Flatten())
    # Capas densas
    for i in range(num_layers_dense):
        model.add(Dense(num_neurons, activation=activation))
    # Capa de salida
    model.add(Dense(1))
    # Compilando el modelo
    model.compile(optimizer=getattr(optimizers, optimizer)(learning_rate=lr),
                  loss=loss, 
                  metrics=metrics)
    # Si no hay paciencia no se pone el callback
    if patience == 0:
        # Entrenando el modelo
        history = model.fit(X, y, 
                            validation_split=val_split,
                            epochs=epochs,
                            verbose=verbose)
    # Si hay paciencia se pone el callback
    else:
        # Entrenando el modelo
        history = model.fit(X, y, 
                            validation_split=val_split,
                            epochs=epochs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
                            verbose=verbose)
    ultimo_estado = np.round(history.history[f"val_{metrics[0]}"][-1], decimals=4)
    # Obteniendo las predicciones
    y_pred = model.predict(X_test)
    # Obteniendo el R2
    r2 = np.round(r2_score(y_test, y_pred), decimals=4)
    # Mostrando lo obtenido
    print(f"El último estado de la métrica {metrics[0]} es: {ultimo_estado}")
    print(f"El R2 del modelo es: {r2}")
    # Crear dataframe con los datos de test y las predicciones
    df_pred = pd.DataFrame({'test': y_test.flatten(), 
                            'pred': y_pred.flatten()}, 
                            index=index)
    # Graficando el historial
    if plot_history:
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        pd.DataFrame(history.history).plot(ax=ax[0])
        df_pred.plot(ax=ax[1])
        df_pred['error'] = df_pred['test'] - df_pred['pred']
        df_pred["error"].plot(ax=ax[2])
        ax[0].set(title="Loss vs Val loss", xlabel=None)
        ax[1].set(title="Prediccion", xlabel=None)
        ax[2].set(title="Error", xlabel=None)
        plt.tight_layout()
    return model, history
