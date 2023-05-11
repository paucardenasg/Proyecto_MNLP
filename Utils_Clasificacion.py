# Librerías
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Funciones
def get_categories(data: pd.DataFrame) -> pd.DataFrame:
    # Crear listas para cada categoría nueva
    fruits = ['Apple', 'Banana', 'Fenugreek', 'Grapes', 'Lime', 'Mango', 'Papaya', 'Parseley', 'Pineapple', 'Pomegranate',
            'Pumpkin', 'Sugarcane', 'Sweet', 'Tomato', 'Guava', 'Mombin', 'Barela', 'Lemon',  'Orange', 'Mandarin',
            'Strawberry',  'Pear', 'Litchi', 'Kiwi', 'Tamarind', 'Kinnow', 'Water', 'Jack']
    veggies = ['Asparagus', 'Brinjal', 'Brocauli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauli', 'Celery', 'Chilli',
                'Christophine', 'Cucumber', 'Garlic', 'Lettuce', 'Mushroom', 'Onion',  'Potato', 'Spinach',
                'Squash', 'Sugarbeet', 'Raddish', 'Maize', 'Cowpea', 'Yam', 'Knolkhol', 'Turnip', 'Red', 'Cow',
                'Brd', 'Sword']
    others = ['Bamboo', 'Bitter', 'Bottle', 'Clive', 'Coriander', 'Cress', 'Drumstick', 'Fish',
            'French', 'Ginger', 'Green', 'Gundruk', 'Mint', 'Mustard', 'Neuro', 'Okara', 'Pointed', 'Smooth',
            'Snake', 'Soyabean', 'Tofu', 'Arum', 'Bakula', 'Bauhania', 'Musk', 'Sponge', 'Fennel']
    # Agregar la clase al DataFrame
    prod_clases = []
    for product in data['Clase']:
        if product in fruits:
            prod_clases.append('1')
        elif product in veggies:
            prod_clases.append('2')
        else:
            prod_clases.append('0')
    # En este caso nuestra variable de salida será la clase
    data['y'] = prod_clases
    # Eliminar las columnas que no importan para el problema
    data = data.drop(['Mínimo', 'Máximo', 'Clase', 'Inflacion', 'Precio_Dolar', 'Desempleo'], axis=1)
    # Retornar el DataFrame
    return data

# Función para graficar las longitudes de las series de tiempo
def plot_length_ts(products_ts: dict):
    # Contabilizar productos por categoría y obtener la longitud de su serie de tiempo
    fruit_data = []
    vegg_data = []
    other_data = []
    # Contabilizar productos por categoría y obtener la longitud de su serie de tiempo
    for values in products_ts.values():
        if values[1] == '2':
            vegg_data.append(len(values[0]))
        elif values[1] == '1':
            fruit_data.append(len(values[0]))
        else:
            other_data.append(len(values[0]))
    # Visualizar la distribución de las longitudes de las series de tiempo según su categoría
    fig, axs = plt.subplots(1, 3)
    # Haciendo más grande la figura
    fig.set_size_inches(15, 5)
    # Graficando
    axs[0].hist(fruit_data, color='darkblue')
    axs[0].set_title('Fruits')
    axs[1].hist(vegg_data, color='teal')
    axs[1].set_title('Veggies')
    axs[2].hist(other_data, color='maroon')
    axs[2].set_title('Others')
    for ax in axs.flat:
        ax.label_outer()
    # Series por categoría
    print(f' + Frutas: {len(fruit_data)} \n + Verduras: {len(vegg_data)} \n + Otros: {len(other_data)}')
    plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Clase para transformar los datos
class TransformData:
    # Constructor
    def __init__(self, products_ts: dict, umbral: int) -> None:
        # Obteniendo una lista con los nombres de los productos
        prods = list(set(products_ts.keys()))
        # Arrays donde pondremos los resultados de X y y
        X = []
        y = []
        # Ciclo para movernos por productos
        for prod in prods:
            # Obteniendo los datos de la serie de tiempo
            temp_ts = products_ts[prod][0]
            # Obteniendo la etiqueta
            temp_label = products_ts[prod][1]
            # Dividiendo en X e y
            X.append(temp_ts)
            y.append(temp_label)
        # Convirtiendo a array
        data = np.array(X)
        labels = np.array(y)
        # Guardando los datos
        self.data = data
        self.labels = labels
        self.umbral = umbral
    
    # Método para saber la longitud de la serie
    def get_len(self, serie):
        return serie.shape[0]
    
    # Método para aplicar el umbral a los datos
    def apply_umbral(self, serie):
        while self.get_len(serie) != self.umbral:
            # Checando si la serie es mayor al umbral
            if self.get_len(serie) > self.umbral:
                # Recortando la serie
                serie = serie[:self.umbral]
            # Checando si la serie es menor al umbral
            elif self.get_len(serie) < self.umbral:
                # Obteniendo la diferencia
                diferencia = self.umbral - self.get_len(serie)
                # Rellenando la serie con una copia de la serie
                serie = np.concatenate((serie, serie[:diferencia]))
        # Regresando la serie
        return serie
        
    # Método para hacer una lista un array
    def list_to_array(self, lista):
        return np.array(lista)
    
    # Método para aplicar el umbral a todos los datos
    def apply_transformation(self):
        new_data = []
        # Ciclo para recorrer los datos
        for serie in self.data:
            # Aplicando el umbral a la serie
            serie = self.apply_umbral(serie)
            if serie.shape[0] != self.umbral:
                print(f'Longitud de la serie diferente al umbral {len(serie)}')
            # Agregando la serie a los nuevos datos
            new_data.append(serie)
        # Cambiando el tipo de dato de los arrays a float32
        new_data = self.list_to_array(new_data).astype('float32')
        self.labels = self.list_to_array(self.labels).astype('float32')
        # Imprimiendo la forma de los nuevos datos
        print(f'Forma de los nuevos datos: {np.array(new_data).shape}')
        print(f'\t - Tenemos {len(new_data)} series de tiempo')
        print(f'\t\t - Cada serie de tiempo tiene {new_data[0].shape[0]} datos')
        # Regresando los nuevos datos
        return np.array(new_data), self.labels
    

# Función para dividir el conjunto de datos en entrenamiento y prueba
def split_data_clasification(data, labels, test_size, shuffle=True, reshape=True):
    # Obteniendo la cantidad de datos de prueba
    n_test = int(data.shape[0] * test_size)
    # Mezclando los datos
    if shuffle:
        indices_aletaorios = np.random.permutation(len(data))
        data = data[indices_aletaorios]
        labels = labels[indices_aletaorios]
    # Obteniendo los datos de entrenamiento
    X_train = data[n_test:]
    y_train = labels[n_test:]
    # Obteniendo los datos de prueba
    X_test = data[:n_test]
    y_test = labels[:n_test]
    # Agregando una dimensión a los datos de y
    if reshape:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    # Imprimiendo las formas de los datos
    print('- Forma de los Datos de Entrenamiento:')
    print(f'\t- Forma de los datos de Entrenamiento: {X_train.shape}')
    print(f'\t- Forma de las etiquetas de Entrenamiento: {y_train.shape}')
    print('- Forma de los Datos de Prueba:')
    print(f'\t- Forma de los datos de Prueba: {X_test.shape}')
    print(f'\t- Forma de las etiquetas de Prueba: {y_test.shape}')
    # Regresando los datos
    return X_train, X_test, y_train, y_test
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Función para obtener las métricas de evaluación
def get_metrics(nombre_modelo, y_true, y_pred):
    accuracy = np.round(accuracy_score(y_true, y_pred), decimals=4)
    recall = np.round(recall_score(y_true, y_pred, average='weighted'), decimals=4)
    precision = np.round(precision_score(y_true, y_pred, average='weighted'), decimals=4)
    f1 = np.round(f1_score(y_true, y_pred, average='weighted'), decimals=4)
    # Creando un Df con las métricas
    df = pd.DataFrame({'Accuracy': [accuracy], 'Recall': [recall], 'Precision': [precision], 'F1': [f1]}).T.reset_index().rename(columns={'index': 'Métricas', 0: 'Valor'})
    df['Modelo'] = nombre_modelo
    # Ordenando el Df
    df = df[['Modelo', 'Métricas', 'Valor']]
    # Regresando el Df
    return df

# Función para graficar la matriz de confusión
def plot_confusion_matrix(y_true, y_pred, labels):
    # Obtener la matriz de confusión
    confusion_mat = confusion_matrix(y_true, y_pred)
    # Crear el mapa de calor de la matriz de confusión
    sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')    
    # Configurar el título y los ejes del mapa de calor
    plt.title('Matriz de confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Verdaderos valores')
    # Añadir las etiquetas de las clases
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # Mostrar el mapa de calor
    plt.show()

# Función para obtener la evaluación del modelo
def get_evaluation(nombre_modelo, y_true, y_pred, labels):
    # Obteniendo las métricas
    df_metrics = get_metrics(nombre_modelo, y_true, y_pred)
    # Obteniendo la matriz de confusión
    plot_confusion_matrix(y_true, y_pred, labels)
    # Regresando el Df
    return df_metrics

# Función para evaluar la agrupación de K-means
def get_evaluation_kmeans(modelo, y_true, y_pred):
    silhouette = np.round(silhouette_score(y_true, y_pred), decimals=4)
    # Df con el resultado
    df = pd.DataFrame({'Coeficiente de Silueta': [silhouette]}).T.reset_index().rename(columns={'index': 'Métrica', 0: 'Valor'})
    df['Modelo'] = modelo
    # Ordenando el Df
    df = df[['Modelo', 'Métrica', 'Valor']]
    return df
