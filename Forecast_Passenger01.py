# -*- coding: utf-8 -*-
"""
FORECAST DEL DATASETS DE PASSENGER 
"""

# LIBRERIAS
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# PATH
path = "C:/Users/USUARIO/Desktop/CursoDataScien/Machine/"

# Leemos los datos del CSV
data_passen = pd.read_csv(path + "airline_passengers.csv", index_col=('Month'), parse_dates=(True))
print(data_passen.head())
print("\n\n")
print(data_passen.info())
print("\n\n")

# 80% train y 20% test, la forma de entrenar los modelos con los datos obtenidos
train_data = data_passen.iloc[:109]
test_data = data_passen.iloc[108:]

# Ajustamos el modelo ya que tienen una tendencia multiplicativa, y su 
# periodicidad/tendencia es por 12 meses
model_fit = ExponentialSmoothing(train_data['Thousands of Passengers'],
                                 trend='mul', # tendencia multiplicativa
                                 seasonal='mul', 
                                 seasonal_periods=12).fit() # 12 meses al año como entrada

### Pronosticamos/Ajustamos 36 periodos futuros
test_predictions = model_fit.forecast(36)
print(test_predictions.head())
print("\n\n")

# Visualizamos en un grafico las datos y prediciones tomadas
fig, ax = plt.subplots(figsize=(12, 8))
train_data['Thousands of Passengers'].plot(ax=ax,legend=True, label='TRAIN')
test_data['Thousands of Passengers'].plot(ax=ax,legend=True, label='TEST', title='Datos del Dataset')
ax.legend();

# Grafico solo de la prediccion a partir del train
fig, ax = plt.subplots(figsize=(13, 8))
test_predictions.plot(ax=ax,legend=True,label='PREDICTION DEL TRAIN', xlim=['1958-01-01','1961-01-01'], title='Prediccion a partir del Train') # Ajustamos el grafico para verlo mas "limitado con el xlim"
ax.legend();

# Grafico mixto desde los datos del train mas su prediccion para comprobar que nuestro modelo se 
# ajusta a nuestro dataset correctamente
fig, ax = plt.subplots(figsize=(13, 9))
train_data['Thousands of Passengers'].plot(ax=ax,legend=True, label='TRAIN')
test_predictions.plot(ax=ax,legend=True,label='PREDICTION DEL TRAIN', title='Train + Prediccion a partir Train') # Ajustamos el grafico para verlo mas "limitado con el xlim"
ax.legend();

# Visualizacion por pantalla de las medias, valor absoluto del error y cuadratico
print(test_data.describe())
print(mean_absolute_error(test_data, test_predictions))
print(mean_squared_error(test_data, test_predictions)) # Valor cuadratico

## Volvemos a entrenar el modelo con suavizado exponentia, ajustando el modelo
# tendencia multiplicativa, y su periodicidad/tendencia es por 12 meses
final_model = ExponentialSmoothing(data_passen['Thousands of Passengers'],
                                   trend='mul',
                                   seasonal='mul',
                                   seasonal_periods=12).fit()

# Prediccion por 3 años seguidos, 12 * 3 = 36
forecast_predic = final_model.forecast(36)

#Visualizamos las predicciones y df de pasajeros
fig, ax = plt.subplots(figsize=(14, 8))
data_passen['Thousands of Passengers'].plot(ax=ax,legend=True, label='Dataset Passenger')
forecast_predic.plot(ax=ax,legend=True, label='Prediccion_Forecast', title = 'DATASET + FORECAST');
ax.legend();

# Se observa que las predicciones crecen igual que la grafica del modelo no estacionario
# Siendo esto una buena suposicion
fig, ax = plt.subplots(figsize=(15, 8))
train_data['Thousands of Passengers'].plot(ax=ax,legend=True, label='TRAIN')
test_data['Thousands of Passengers'].plot(ax=ax,legend=True, label='TEST')
test_predictions.plot(ax=ax,legend=True,label='PREDICTION DEL TRAIN') 
forecast_predic.plot(ax=ax,legend=True, label='Prediccion_Forecast')
ax.legend();








