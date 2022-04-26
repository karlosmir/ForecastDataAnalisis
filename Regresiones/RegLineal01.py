# -*- coding: utf-8 -*-
"""
REGRESION LINEAl SIMPLE DE SALARIO/EXPERIENCIA
"""
import pandas as pd
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# PATH
path = "C:/Users/USUARIO/Desktop/CursoML/Data/"
data_salary = pd.read_csv(path + "Salary_Data.csv")
#print(data_salary.info())  # Son 30 datos
#print(data_salary.head())
X = data_salary.iloc[:, :-1].values
#print(X)
Y = data_salary.iloc[:, 1].values
#print(Y)

# Dividimos los datos usando la funcion train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=(0))

# Modelo de Regresion lineal
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
y_pred = regression.predict(X_test)
#print("Datos del TEST\n" +  str(y_test))
#print("Datos del PRED\n" +  str(y_pred))

plt.scatter(X_train, y_train, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("Sueldo VS Años de Experiencia")
plt.xlabel("Años de Eperiencia")
plt.ylabel("Sueldo")
plt.show();






























