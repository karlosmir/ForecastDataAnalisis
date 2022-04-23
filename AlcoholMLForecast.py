"""
FORECAST DEL DATASETS DE ALCOHOL SALES 
"""

# LIBRERIAS
from DickeyFuller import adf_test
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# PATH
path = "C:/Users/USUARIO/Desktop/CursoDataScien/Machine/"

# Leemos los datos del CSV
data_alcohol = pd.read_csv(path + "Alcohol_Sales.csv", index_col=('DATE'), parse_dates=(True))
print(data_alcohol.head())
print("\n\n\n")
print(data_alcohol.info())
print("\n\n\n")

# Grafica
nombre = adf_test(data_alcohol)
fig, ax = plt.subplots(figsize=(12, 8))
data_alcohol.plot(ax=ax, legend=True, title=str(nombre))
ax.set(xlabel='Date', ylabel='NumSales')
plt.show();

# Preparamos las variables para entrenar y testear nuestro modelo
train = data_alcohol.iloc[:261]
test = data_alcohol.iloc[261:]

model_fit = ExponentialSmoothing(train,
                                 trend='mul', # tendencia multiplicatica
                                 seasonal='mul', 
                                 seasonal_periods=12).fit()

### Pronosticamos/Ajustamos 36 periodos futuros
test_predictions = model_fit.forecast(100)
print("\n\n\n")
print(test_predictions.head())
print("\n\n\n")

# Visualizamos en un grafico las datos del train y test
fig, ax = plt.subplots(figsize=(12, 8))
train['S4248SM144NCEN'].plot(ax=ax,legend=True, label='TRAIN')
test['S4248SM144NCEN'].plot(ax=ax,legend=True, label='TEST', title='Datos del Dataset')
ax.legend();

# Grafico solo de la prediccion a partir del train
fig, ax = plt.subplots(figsize=(13, 8))
test_predictions.plot(ax=ax,legend=True,label='PREDICTION DEL TRAIN', xlim=['2019-01-01','2021-01-01'], title='Prediccion a partir del Train') # Ajustamos el grafico para verlo mas "limitado con el xlim"
ax.legend();

# Grafico mixto desde los datos del train mas su prediccion para comprobar que nuestro modelo se 
# ajusta a nuestro dataset correctamente
fig, ax = plt.subplots(figsize=(13, 9))
train['S4248SM144NCEN'].plot(ax=ax,legend=True, label='TRAIN')
test_predictions.plot(ax=ax,legend=True,label='PREDICTION DEL TRAIN', title='Train + Prediccion a partir Train') 
ax.legend();

## Volvemos a entrenar el modelo con suavizado exponentia, ajustando el modelo
# tendencia multiplicativa, y su periodicidad/tendencia es por 12 meses
final_model = ExponentialSmoothing(data_alcohol,
                                   trend='mul',
                                   seasonal='mul',
                                   seasonal_periods=12).fit()

# Prediccion por 3 años seguidos, 12 * 3 = 36
forecast_predic = final_model.forecast(36)

#Visualizamos las predicciones y df de pasajeros
fig, ax = plt.subplots(figsize=(14, 8))
data_alcohol['S4248SM144NCEN'].plot(ax=ax,legend=True, label='Dataset Alcohol SALES')
forecast_predic.plot(ax=ax,legend=True, label='Prediccion_Forecast', title = 'DATASET + FORECAST');
ax.legend();

# Se observa que las predicciones crecen igual que la grafica del modelo no estacionario
# Siendo esto una buena suposicion
fig, ax = plt.subplots(figsize=(15, 8))
train['S4248SM144NCEN'].plot(ax=ax,legend=True, label='TRAIN')
test['S4248SM144NCEN'].plot(ax=ax,legend=True, label='TEST')
test_predictions.plot(ax=ax,legend=True,label='PREDICTION DEL TRAIN') 
forecast_predic.plot(ax=ax,legend=True, label='Prediccion_Forecast')
ax.legend();













