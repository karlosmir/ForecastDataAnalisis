"""
REGRESION LINEAl MULTIPLE
Datasets de 50 empresas con datos geograficos
con financiacion de compañias para realizar startups
con el gasto de IDE, estado geografico, beneficio, etc
Encontrar la Correlacion
Ganancia empresa = ganancia0
y = b0 + b1*X1..bn*Xn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm # Coeficiente del termino independiente

# PATH
path = "C:/Users/USUARIO/Desktop/CursoML/Data/"
data_startup = pd.read_csv(path + "50_Startups.csv")
#print(data_startup.info())  # Son 50 datos
#print(data_startup.head())
X = data_startup.iloc[:, :-1].values
#print(X)
Y = data_startup.iloc[:, 4].values
#print(Y)



# Pasar los datos State de string a variables Categoricas siendo Numericos
# MAS NUEVO COLUMTRANSFORMER
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Evita la trampa de las variables ficticias
X = X[:,1:]

# Dividimos los datos usando la funcion train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=(0))

# Ajustamos el modelo de regresion lineal multiple con el conjunto de TRAIN
regression = LinearRegression()
regression.fit(X_train, y_train)

# Prediccion de los resultados
y_pred = regression.predict(X_test)

# ELIMINACION HACIA ATRAS , Mirar Transparencia PASOS
X = np.append(arr=np.ones((50,1)).astype(int), values = X, axis=1) # 50 filas en 1 columna de unos
X_opt = X[:, [0,1,2,3,4,5]].tolist()
SL = 0.5 # Nivel de significacion

# Tecnica de los minimos cuadrados OLS para el modelo de regression lineal multiple
regressor_OLS = sm.OLS(Y, X.tolist()).fit()
# Observamos cual es la variable con p_value mas grande y la eliminamos de nuestro datos siguiendo
# la transparencia y luego volvemos a ajustar el DF
print(regressor_OLS.summary())
# Eliminamos la variable X2 por ser la mas alta y hacemos lo mismo de antes sin la column 2
X_opt = X[:, [0,1,3,4,5]].tolist()
regressor_OLS = sm.OLS(Y, X_opt).fit()
print(regressor_OLS.summary())
# Eliminamos la variable x1 por ser la siguiente mas alta
X_opt = X[:, [0,3,4,5]].tolist()
regressor_OLS = sm.OLS(Y, X_opt).fit()
print(regressor_OLS.summary())
# Eliminamos la variable x2 por ser la siguiente mas alta
# Ya que es mayor que 0.5, SL = 0.5 # Nivel de significacion
X_opt = X[:, [0,3,5]].tolist()
regressor_OLS = sm.OLS(Y, X_opt).fit()
print(regressor_OLS.summary())
# Eliminamos la variable x2 otra vez
X_opt = X[:, [0,3]].tolist()
regressor_OLS = sm.OLS(Y, X_opt).fit()
print(regressor_OLS.summary())
# Con el criterio bayesiando o akaike es mejor opcion
# para saber que criterio utilizar para mejor decision

# Este modelo es regresion lineal simple clasica siendo el mejor modelo
# precisor es el gasto en IDE, ya que tiene un gasto importante en el
# beneficio de la empresa, una variable predictora 

plt.scatter(X_train[:,2] , y_train, color='red')  #X_train[2] IDE
plt.plot(X_test[:,2], y_pred, color='green')
plt.title('Regresion lineal Multiple a Simple ')
plt.xlabel('Valores independientes IDE)')
plt.ylabel('Beneficio')
plt.show()


















