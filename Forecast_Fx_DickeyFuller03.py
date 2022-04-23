# AIC = Criterio de informacion Akaike (estimador del error de predicción)
# LIBRERIAS
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import month_plot, quarter_plot
import warnings
warnings.filterwarnings('ignore')

############## FUNCION DICKEY FULLER TEST ##############
# Te dice si el modelo es estaionario o no ya que verifica automaticamente
# el indice del resultado del valor p-value de modo que asi teniendo una
# fuerte hipotesis o una debil hipotesis
def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
        x = True
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        x = False
        
    if x:
        nombre="Modelo Estacionario"
    else:
        nombre="Modelo No Estacionario"
    return nombre
 
#PATH
path = "C:/Users/USUARIO/Desktop/CursoDataScien/Machine/"

# MODELO NO ESTACIONARIO
data_passen = pd.read_csv(path + "airline_passengers.csv", index_col=('Month'), parse_dates=(True)) 
data_passen.index.freq = 'MS'
# MODELO ESTACIONARIO
data_birth = pd.read_csv(path + "DailyTotalFemaleBirths.csv", index_col=('Date'), parse_dates=(True)) 
data_birth.index.freq = 'D'

# Prueba funcion aumentada de Dicky Fuller que devuelve el valor de estadistico de P utilizando retrasos
datatest_passen= adfuller(data_passen['Thousands of Passengers'])
dataout_passen = pd.Series(datatest_passen[0:4], index=['ADF TEST STATICS', 'p-value','Lags Used','Observations'])
print(dataout_passen)

for key,val in datatest_passen[4].items():
    dataout_passen[f'crital value ({key})'] = val
print("\n\n")
print(dataout_passen)
print("\n\n")

# Usamos la funcion Dickey definida arriba
nombre = adf_test(data_passen['Thousands of Passengers'])
fig, ax = plt.subplots(figsize=(12, 8))
data_passen['Thousands of Passengers'].plot(ax=ax,legend=True, title=str(nombre))
ax.legend();

print("\n\n")
nombre = adf_test(data_birth['Births'])
fig, ax = plt.subplots(figsize=(14, 7))
data_birth['Births'].plot(ax=ax,legend=True, title=str(nombre))
ax.legend()

### Funciones monthplot y quarterplot
month_plot(data_passen['Thousands of Passengers'])
dfq = data_passen['Thousands of Passengers'].resample(rule='Q').mean()
quarter_plot(dfq) # Por trimestres interpretamos que los viajeros viajan mas en 3ºtrimestre






























