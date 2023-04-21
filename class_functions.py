import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler

class LinearForecast:

    def __init__(self, data):
        self.data = data

    def transform_series(self, method):
        series = self.data

        if method == 'log':
            transformed_series = pd.Series(np.log(series), index=series.index)
        elif method == 'boxcox':
            pt = PowerTransformer(method='box-cox')
            transformed_series = pd.Series(pt.fit_transform(series.values.reshape(-1, 1)).squeeze(), index=series.index)
        elif method == 'yeojohnson':
            pt = PowerTransformer(method='yeo-johnson')
            transformed_series = pd.Series(pt.fit_transform(series.values.reshape(-1, 1)).squeeze(), index=series.index)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            transformed_series = pd.Series(scaler.fit_transform(series.values.reshape(-1, 1)).squeeze(), index=series.index)
        elif method == 'standard':
            scaler = StandardScaler()
            transformed_series = pd.Series(scaler.fit_transform(series.values.reshape(-1, 1)).squeeze(), index=series.index)
        else:
            raise ValueError('Método de transformación no válido.')
        
        transformed_series = self.data
        return transformed_series
    
    @staticmethod
    def evaluate_forecasts(actual, predicted):
        """
        Función que se encargará de calcular las métricas de error. (MSE, RMSE, MAD, MAPE)
        :param actual: Valor actual de la serie de tiempo que se desea predecir.
        :param predicted: Valor predicho por el modelo ajustado.
        """
        # Mean Squared Error (MSE)
        mse_metric = np.mean((predicted - actual) ** 2)
        # Root Mean Squared Error (RMSE)
        rmse_metric = np.sqrt(mse_metric)
        # Mean Absolute Percentage Error (MAPE)
        mape_metric = np.mean(np.abs((actual - predicted) / actual)) * 100
        # Mean Absolute Deviation (MAD)
        mad_metric = np.mean(np.abs(predicted - actual))
        return mse_metric, rmse_metric, mape_metric, mad_metric


    def split_dataset(self, train_size: float):
        """
        Función que se encarga de dividir la data ingresada en datos de entrenamiento y prueba.
        :param data: Timeseries
        :param train_size: Tamaño de la division del dataset de entrenamiento (0, 1).
        """
        data = self.data
        split = int(np.round(data.shape[0] * train_size))
        train_data = data[:split]
        test_data = data[split:]
        return train_data, test_data
    

    def plot_acf_pacf(self, kwargs: dict):
        """
        Función que grafica las funciones de autocorrelación y autocorrelación parcial,
        depende de matplotlib y plot_acf, plot_pacf (statsmodels).
        :param data: Timeseries
        :param kwargs: Argumentos de las funciones de statsmodels
        :return: Gráficas de acf y pacf
        """
        data = self.data
        f = plt.figure(figsize=(8, 5))
        ax1 = f.add_subplot(121)
        plot_acf(data, zero=False, ax=ax1, **kwargs)
        ax2 = f.add_subplot(122)
        plot_pacf(data, zero=False, ax=ax2, method='ols', **kwargs)
        plt.show()


    def adf_test(self):
        """
        Se calcula si la serie de tiempo es estacionaria con el test de "Dickey-Fuller"
        :return: Simplemente un "print" que nos dice si la serie es estacionaria o no.
        """
        data = self.data
        print("Results of Dickey-Fuller Test:")
        dftest = adfuller(data, autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        print(dfoutput)
        if (dftest[1] <= 0.05) & (dftest[4]['5%'] > dftest[0]):
            print("\u001b[32mStationary\u001b[0m")
        else:
            print("\x1b[31mNon-stationary\x1b[0m")

    @staticmethod
    def prediction(timeseries, size, clf, dyn: False):
        end = len(timeseries)
        start = end - size
        actual = timeseries[start:end]
        actual = actual.to_numpy()
        predicted = clf.predict(start=start + 1, end=end, dynamic=dyn)
        actual = actual.reshape(size, )
        assert actual.shape == predicted.shape

        predictions = pd.DataFrame({'actual': actual, 'predicted': predicted}, index=timeseries.index[start:end])

        fig, ax = plt.subplots(figsize=(12, 8))
        predictions.actual.plot(ax=ax)
        predictions.predicted.plot(ax=ax)
        ax.legend(labels=['actual', 'forecast'])
        plt.show()

    @staticmethod
    def evaluate_model(test, pred):
        """
        Función que se encargará de evaluar el modelo y calcular las métricas de error.
        :param pred:
        :param test: Datos de prueba.
        """
        mse_metric, rmse_metric, mape_metric, mad_metric = LinearForecast().evaluate_forecasts(test, pred)
        return mse_metric, rmse_metric, mape_metric, mad_metric

    @staticmethod
    def arima_model(data, ar, diff, ma):
        """
        Ejecución del modelo ARIMA
        :param data: data a usar, no se usa del constructor para hacer diferenciaciones más fácil.
        :param ar: Puede ser tupla o entero de parámetros de la parte autorregresiva.
        :param diff: Número de diferenciaciones.
        :param ma: Puede ser tupla o entero de parámetros de la parte media móvil.
        :return: Modelo entrenado
        """
        arima_model = ARIMA(data, order=(ar, diff, ma))
        model = arima_model.fit()
        return model

    @staticmethod
    def decompose_timeseries_stl(data, period: int, seasonal_deg: int, seasonal, residual=False):
        seasonal = STL(data, period=period, seasonal_deg=seasonal_deg, seasonal=seasonal)
        res = seasonal.fit()
        res.plot(resid=residual, observed=True)
        plt.show()
        return res

    @staticmethod
    def decompose_timeseries_mstl(data1, periods_seasonality: tuple, stl_kwargs: dict):
        """
        Función para graficar y descomponer una serie de tiempo
        :return: Serie de tiempo descompuesta
        """
        model = MSTL(data1, periods=periods_seasonality, stl_kwargs=stl_kwargs)
        result = model.fit()
        # Gráfica de descomposición
        fig, ax = plt.subplots(len(periods_seasonality) + 3, 1, sharex="all", figsize=(8, 8))
        conteo_axis = 0
        # Gráfica normal
        result.observed.plot(ax=ax[0])
        ax[conteo_axis].set_ylabel('Observed')
        conteo_axis += 1
        # Gráfica tendencia
        result.trend.plot(ax=ax[1])
        ax[conteo_axis].set_ylabel('Trend')
        conteo_axis += 1
        # Ciclo para agregar todas las estacionalidades encontradas
        for i in range(len(periods_seasonality)):
            result.seasonal[f'seasonal_{periods_seasonality[i]}'].plot(ax=ax[conteo_axis])
            ax[conteo_axis].set_ylabel(f'seasonal_{periods_seasonality[i]}')
            conteo_axis += 1
        # Gráfica del residual
        result.resid.plot(ax=ax[conteo_axis])
        ax[conteo_axis].set_ylabel('Residual')
        fig.tight_layout()
        plt.show()
        return result

    @staticmethod
    def sarimax_params(p: list, d: list, q: list, P: list, D: list, Q: list, t: list, s: int):
        """
        Función que genera todas las posibles combinaciones para los modelos sarimax
        :param p: Lista con todos los hiperparámetros.
        :param d: Lista con todos los hiperparámetros.
        :param q: Lista con todos los hiperparámetros.
        :param P: Lista con todos los hiperparámetros.
        :param D: Lista con todos los hiperparámetros.
        :param Q: Lista con todos los hiperparámetros.
        :param t: Lista con todos los hiperparámetros.
        :param s: Lista con todos los hiperparámetros.
        :return: Lista con todas las combinaciones.
        """
        param_no_estacionales = list(itertools.product(p, d, q))
        param_estacionales = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q))]
        sarimax_params = list(itertools.product(param_no_estacionales, param_estacionales, t))
        return sarimax_params

    @staticmethod
    def training_sarimax_multiple_params(data, sarimax_params):
        """
        Función para ejecutar multiples modelos sarimax. (no se usa la timeseries del constructor
         por si se desea hacer transformaciones).
        :param data: Data con la serie de tiempo
        :param sarimax_params: Lista con las posibles combinaciones para el sarimax.
        :return: Dataframe ordenado para ver el mejor modelo.
        """
        resultados = {"params": [], "AIC": [], "BIC": [], "LLF": []}

        for par_no_esta, par_esta, trend in sarimax_params:
            mod = SARIMAX(
                endog=data,
                trend=trend,
                order=par_no_esta,
                seasonal_order=par_esta
            )
            results = mod.fit(disp=False)
            resultados["params"].append(str((par_no_esta, par_esta, trend)))
            resultados["AIC"].append(results.aic)
            resultados["BIC"].append(results.bic)
            resultados["LLF"].append(results.llf)
        resultados = (pd.DataFrame(resultados)
                        .sort_values(by=['AIC'], ascending=True)
                        .reset_index(drop=True))
        return resultados

    @staticmethod
    def sarimax_model(data, trend, params_no_est, params_est):
        """
        Modelo para ejecutar el modelo final sarimax
        :param data: Serie de tiempo final.
        :param trend: Parámetro de tendencia.
        :param params_no_est: Tupla de parámetros no estacionarios.
        :param params_est: Tupla de parámetros estacionarios.
        :return: Modelo entrenado
        """
        mod = SARIMAX(
            endog=data,
            trend=trend,
            order=params_no_est,
            seasonal_order=params_est
        )
        results = mod.fit(disp=False)
        return results
