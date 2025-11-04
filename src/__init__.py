# Importaciones de utils
from utils import (load_data, processed_data, split_data,
                   save_model, plot_class_distribution, plot_confusion_matrix)

# Importaciones de librerías
import pandas as pd
import numpy as np
import os

# Importaciones de Sklearn (Modelo y Métricas)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- 0. Definir constantes (Estilo Cookiecutter) ---
DATA_URL = 'https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv'
MODEL_PATH = 'models/sentiment_model_nb.pkl'
REPORTS_PATH = 'reports/figures/'

# --- 1. Cargar datos ---
df = load_data(DATA_URL)

# --- 2. Preparar datos ---
data_processed = processed_data(df)

# --- 3. EDA Básico (Guardar gráfico de distribución) ---
# (Paso lógico de EDA antes de entrenar)
plot_class_distribution(data_processed, 'polarity', os.path.join(REPORTS_PATH, 'distribucion_clases.png'))

# --- 4. Dividir datos ---
X_train, X_test, y_train, y_test = split_data(data_processed, 'polarity')

# --- 5. Inicializar y entrenar el modelo (Pipeline) ---
# (Como se vio en explore.ipynb, el Pipeline con CountVectorizer fue el mejor)
print("Iniciando entrenamiento del Pipeline (CountVectorizer + Naive Bayes)...")

model_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# Entrenamos
model_pipeline.fit(X_train, y_train)
print("Modelo entrenado correctamente")

# --- 6. Realizar predicciones ---
y_pred = model_pipeline.predict(X_test)

# --- 7. Evaluar el modelo ---
print("Evaluando modelo...")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nReporte de Clasificación:")
print(report)

# --- 8. Guardar Gráfico de Evaluación (Matriz de Confusión) ---
plot_confusion_matrix(y_test, y_pred, os.path.join(REPORTS_PATH, 'matriz_confusion_nb.png'))

# --- 9. Guardar modelo ---
save_model(model_pipeline, MODEL_PATH)

print("\n--- Proceso completado ---")