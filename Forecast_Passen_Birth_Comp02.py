"""
AutoCorrelacion, Parcial Autocorrelacion para modelos no estacionarios y estacionarios
Datasets de pasajeros, nacimientos y poblacion estimada en US, 3 ejemplos distintos de  modelos
para comparacion de modelos
"""

# LIBRERIAS
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

#PATH
path = "C:/Users/USUARIO/Desktop/CursoDataScien/Machine/"

# MODELO NO ESTACIONARIO
data_passen = pd.read_csv(path + "airline_passengers.csv", index_col=('Month'), parse_dates=(True)) 
data_passen.index.freq = 'MS'
print(data_passen.head())
#print(data_passen.isnull().sum())


# MODELO ESTACIONARIO
data_birth = pd.read_csv(path + "DailyTotalFemaleBirths.csv", index_col=('Date'), parse_dates=(True)) 
data_birth.index.freq = 'D'
print(data_birth.head())
#print(data_birth.isnull().sum())

# Diagrama de retraso, visualiza una correlacion fuerte entre X e Y 
# Para mostrar uno u otro comentalos, fallo libreria#
fig, ax = plt.subplots(figsize=(8,9))
pd.plotting.lag_plot(data_birth['Births'], ax = ax) # Estacionario
ax.set(title='Births')
ax.legend()
plt.show();

fig, ax = plt.subplots(figsize=(12,9))
pd.plotting.lag_plot(data_passen['Thousands of Passengers']); # no estacionario
ax.set(title='Pasajeros')
ax.legend();
plt.show();

## Diagrama de Correlacion automatica , se observan picos que aumentan anualmente
# la region sombreada representa el 95% y los valores fuera de ese intervalo
# tienen mucha probabilidad de ser una correlacion
# modelo no estacionario
data_passen.plot(title='Dataset Pasajeros');
plot_acf(data_passen, lags=40); # Retrasos laggs 

# En este diagrama existe una caida muy brusca ya que es estacionario, no es tan 
#interpretable como el modelo no estacionario que se ve a la vista
# modelo  estacionario
data_birth.plot();
plot_acf(data_birth, lags=40); # 40 unnidades de Retrasos = xlaggs 

## graficos parciales de correlacion automatica para datos estacionarios
plot_pacf(data_birth, lags=40);

#### MODELO ARIMMA 
### Permite autoajustar de una manera rapida y automatica las predicciones del 
### modelo con la Autoregresion
data_pob = pd.read_csv(path + "uspopulation.csv", index_col=('DATE'), parse_dates=(True)) 
print(data_pob.head())
print(data_pob.describe())
print(len(data_pob))
#print(data_pob.isnull().sum())
data_pob.plot(title='Poblacion US');

# 80% train  y 20% test
train_pob = data_pob.iloc[:84]
test_pob = data_pob.iloc[84:]

# MODELO TRAIN
### Cambia funcion AR a AutoReg de la libreria
modelo = AutoReg(train_pob['PopEst'],lags=1)
ARfit = modelo.fit()

# Definimos el tamaño de nuestra prediccion
start = len(train_pob)
end = len(train_pob) + len(test_pob) - 1

# Donde queremos que empieze y donde acabe la prediccion
pred01 = ARfit.predict(start=start,end=end)
pred01 = pred01.rename('AR(1) PREDICTIONS')

# Visualizacion
test_pob.plot(figsize=(12,8),legend=True)
pred01.plot(legend=True, title='Prediccion 01 del Poblacion US');

# MODELO TEST
modelo2 = AutoReg(train_pob['PopEst'],lags=2)
ARfit2 = modelo2.fit()
pred02 = ARfit2.predict(start=start,end=end)
pred02 = pred02.rename('AR(2) PREDICTIONS')

test_pob.plot(figsize=(12,8),legend=True)
pred01.plot(legend=True)
pred02.plot(legend=True,title='Prediccion 01+02 del Poblacion US');

### MODELO lags 08 con ic = 't-stat'
modelo3 = AutoReg(train_pob['PopEst'],lags=8)
ARfit3 = modelo3.fit()
pred03 = ARfit3.predict(start=start,end=end)
pred03 = pred03.rename('AR(3) PREDICTIONS')

# GRAFICA
fig, ax = plt.subplots(figsize=(12, 8))
test_pob.plot(ax=ax, label='TEST_POB')
pred01.plot(ax=ax, legend=True)
pred02.plot(ax=ax,legend=True)
pred03.plot(ax=ax,legend=True,title='Prediccion 01+02+03 del Poblacion US')
ax.legend();

########## Predecir la poblacion de US para 2023 con forecast
modelo_nuevo = AutoReg(data_pob['PopEst'],lags=1)
ARfit_nuevo = modelo_nuevo.fit()

# Definimos el tamaño de nuestra prediccion, desde donde empezara (final de los
# datos que tenemos y donde queremos que termine) que sera dentro de 3-4 años ya
# que los datos se cogen desde 2011 a 2019 por tanto 12 meses * 4 = 48 
forcast_values = ARfit_nuevo.predict(start=len(data_pob),end=len(data_pob)+48).rename('Forecast')

# GRAFICA
fig, ax = plt.subplots(figsize=(12, 8))
data_pob['PopEst'].plot(ax=ax, legend=True)
forcast_values.plot(ax=ax, legend=True)
ax.legend();





























