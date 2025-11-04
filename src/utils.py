import pandas as pd
import numpy as np
import re
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Funciones para cargar datos y preparar datos

def load_data(url):
    """

    :param url: Url con la data .csv
    :return: un dataframe de pandas
    """
    print(f"Cargando datos desde la URL: {url}")
    df = pd.read_csv(url)
    print(f"Datos cargados correctamente")
    return df


# Función de limpieza y preparación de datos (Adaptada para NLP)

def processed_data(df):
    """

    :param df: Dataframe de pandas (leído del CSV de reviews)
    :return: Data frame de pandas limpio para el modelo (con 'review_clean')
    """
    print("Iniciando limpieza de datos...")
    # Realizo una copia del data frame
    data_processed = df.copy()
    print(f"Copia del dataframe realizada correctamente")

    # Eliminar columnas innecesarias (como en el notebook)
    if 'package_name' in data_processed.columns:
        data_processed.drop('package_name', axis=1, inplace=True)
        print("Columna 'package_name' eliminada")

    # Limpieza de texto
    print("Limpiando columna 'review'")
    # Convertir a minúsculas
    data_processed['review_clean'] = data_processed['review'].str.lower()
    # Eliminar caracteres no alfabéticos (manteniendo espacios)
    data_processed['review_clean'] = data_processed['review_clean'].apply(lambda x: re.sub(r'[^a-z\\s]', ' ', str(x)))
    print("Limpieza de texto completada")

    # Busca si hay duplicados y los elimina
    print(f"Buscando duplicados")
    if data_processed.duplicated().sum() > 0:
        data_processed.drop_duplicates(inplace=True)
        print(f"Duplicados eliminados")

    # Manejar nulos en el texto limpio (si los hubiera)
    if data_processed['review_clean'].isnull().sum() > 0:
        data_processed.dropna(subset=['review_clean'], inplace=True)
        print("Filas con reviews nulas eliminadas")

    return data_processed


def split_data(df, target_column):
    """

    :param df: DataFrame listo para el modelo
    :param target_column: Variable objetivo y (ej: 'polarity')
    :return: data dividida en entrenamiento y prueba (X_train, X_test, y_train, y_test)
    """
    print(f"Dividiendo datos (target={target_column})...")

    # Selecciono la data para X (la columna de texto limpia)
    X = df['review_clean']

    # Selecciono solo la variable objetivo
    y = df[target_column]

    # Utilizo train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Datos divididos en entrenamiento (80%) y prueba (20%)")

    return X_train, X_test, y_train, y_test


def save_model(model, filepath):
    """

    :param model: modelo entrenado (ej: el Pipeline)
    :param filepath: ruta de guardado (ej: 'models/mi_modelo.pkl')
    :return: Guarda el modelo.pkl en la ruta indicada
    """
    print(f"Guardando modelo en: {filepath}")
    # Asegurar que el directorio exista (Estilo Cookiecutter)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print("Modelo guardado correctamente")
    return


# Funciones para graficas

def plot_class_distribution(df, target_column, save_path):
    """
    Genera y guarda el gráfico de distribución de clases (EDA).

    :param df: DataFrame procesado
    :param target_column: Columna objetivo (ej: 'polarity')
    :param save_path: Ruta para guardar el gráfico (ej: 'reports/figures/distribucion_clases.png')
    """
    print("Graficando distribución de clases (EDA)...")
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_column, data=df)
    plt.title('Distribución de Sentimiento (0=Negativo, 1=Positivo)')

    # Asegurar que el directorio exista (Estilo Cookiecutter)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    print(f"Gráfico de distribución guardado en: {save_path}")
    plt.close()  # Cierro para que no se muestre en el script


def plot_confusion_matrix(y_test, y_pred, save_path):
    """
    Genera y guarda la matriz de confusión (Evaluación).

    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :param save_path: Ruta para guardar el gráfico (ej: 'reports/figures/matriz_confusion.png')
    """
    print("Graficando matriz de confusión...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred. Negativo (0)', 'Pred. Positivo (1)'],
                yticklabels=['Real Negativo (0)', 'Real Positivo (1)'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión (CountVectorizer + NB)')

    # Asegurar que el directorio exista
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    print(f"Gráfico de matriz de confusión guardado en: {save_path}")
    plt.close()