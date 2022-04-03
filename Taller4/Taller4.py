# -- coding: utf-8 --
"""
Created on Thu Mar 31 19:41:26 2022

@author: Sebastian Bravo, Sergio Lavao,Daniel Zambrano

Solucion de taller 4
"""
import numpy as np # manejo de matrices
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

# Importar datos de cada clase
rock= np.genfromtxt('data/Punto 1/0.csv', delimiter=',')
scissors= np.genfromtxt('data/Punto 1/1.csv', delimiter=',')
papers= np.genfromtxt('data/Punto 1/2.csv', delimiter=',')
ok= np.genfromtxt('data/Punto 1/3.csv', delimiter=',')

# Segmentación de datos de datos en entrenamiento (80%) y prueba (20%)
Training_0, Testing_0 = model_selection.train_test_split(rock,test_size = int(0.2*len(rock)),train_size = int(0.8*len(rock)))
Training_1, Testing_1 = model_selection.train_test_split(scissors,test_size = int(0.2*len(scissors)),train_size = int(0.8*len(scissors)))
Training_2, Testing_2 = model_selection.train_test_split(papers,test_size = int(0.2*len(papers)),train_size = int(0.8*len(papers)))
Training_3, Testing_3 = model_selection.train_test_split(ok,test_size = int(0.2*len(ok)),train_size = int(0.8*len(ok)))

# Creación matriz de entrenamiento y etiquetas matriz de entrenamiento.
T_m = np.concatenate((Training_0,Training_1,Training_2,Training_3), axis = 0)
y_train = T_m[:,64]
T_m = T_m[:,:-1] # Remoción de etiquetas

# Creación matriz de prueba y etiquetas matriz de prueba.
testing_matrix = np.concatenate((Testing_0,Testing_1,Testing_2,Testing_3), axis = 0)
y_test =  testing_matrix[:,64]
testing_matrix = testing_matrix[:,:-1] # Remoción de etiquetas

#Construccion de matriz de entrenamiento y validacion (Validación (20% del total de datos o 25 % del conjunto de entrenamiento))
train_0, val_0 = model_selection.train_test_split(Training_0, test_size= int(0.25*len(Training_0)), train_size = int(0.75*len(Training_0)))
train_1, val_1 = model_selection.train_test_split(Training_1, test_size= int(0.25*len(Training_1)), train_size = int(0.75*len(Training_1)))
train_2, val_2 = model_selection.train_test_split(Training_2, test_size= int(0.25*len(Training_2)), train_size = int(0.75*len(Training_2)))
train_3, val_3 = model_selection.train_test_split(Training_3, test_size= int(0.25*len(Training_3)), train_size = int(0.75*len(Training_3)))

training_matrix = np.concatenate((train_0,train_1,train_2,train_3), axis = 0)
training_matrix = training_matrix[:,:-1] # Remoción de etiquetas
validating_matrix = np.concatenate((val_0,val_1,val_2,val_3), axis = 0)
validating_matrix = validating_matrix[:,:-1] # Remoción de etiquetas

# Etiquetas para entrenamiento y validacion
y_train = np.concatenate((np.zeros(len(train_0)), np.ones(len(train_1)), 2*np.ones(len(train_2)) , 3*np.ones(len(train_3))), axis = 0)
y_val = np.concatenate((np.zeros(len(val_0)), np.ones(len(val_1)), 2*np.ones(len(val_2)) , 3*np.ones(len(val_3))), axis = 0)

# Comprobación homogeneidad de datos
y_train_0 = len(np.where(y_train == 0)[0]) 
y_train_1 = len(np.where(y_train == 1)[0]) 
y_train_2 = len(np.where(y_train == 2)[0]) 
y_train_3 = len(np.where(y_train == 3)[0]) 

# NOTA: LOS DATOS NO SON HOMOGENEOS, PERO LA DIFERENCIA ENTRE EL NÚMERO DE DATOS ENTRE CLASES ES MENOR AL 1.5%

class_weights = {0:y_train_0/(y_train_1+y_train_2+y_train_3+y_train_0),
                1:y_train_1/(y_train_1+y_train_2+y_train_3+y_train_0),
                2:y_train_2/(y_train_1+y_train_2+y_train_3+y_train_0),
                3:y_train_3/(y_train_1+y_train_2+y_train_3+y_train_0),}

# Entrenamiento de SVM polinomial
# F1 score por iteración
SVM_OVA_F1_scores = []
SVM_OVO_F1_scores = []

for i in range(1,11):
    #Construcción del modelo con transformación OVA
    SVM_OVA = svm.SVC(gamma = 'auto',
                      degree = i,
                      kernel = 'poly',
                      class_weight = class_weights,
                      decision_function_shape='ovr',
                      verbose=1)

    #Construcción del modelo con transformación OVO
    SVM_OVO = svm.SVC(gamma = 'auto',
                      degree = i,
                      kernel = 'poly',
                      class_weight = class_weights,
                      decision_function_shape='ovo',
                      verbose=1)

    # Entrenamiento de modelos
    SVM_OVA.fit(training_matrix, y_train)
    SVM_OVO.fit(training_matrix, y_train)

    # Validación
    y_out = SVM_OVA.predict(validating_matrix) 
    SVM_OVA_F1_scores.append(100*f1_score(y_val, y_out, average = 'weighted'))

    y_out = SVM_OVO.predict(validating_matrix) 
    SVM_OVO_F1_scores.append(100*f1_score(y_val, y_out, average = 'weighted'))

# NOTA: EL mejor resultado se obtuvo con polinomios de grado 2

SVM_OVA = svm.SVC(gamma = 'auto',
                  degree = 2,
                  kernel = 'poly',
                  class_weight = class_weights,
                  decision_function_shape='ovr',
                  verbose=1)

#Construcción del modelo con transformación OVO
SVM_OVO = svm.SVC(gamma = 'auto',
                  degree = 2,
                  kernel = 'poly',
                  class_weight = class_weights,
                  decision_function_shape='ovo',
                  verbose=1)

# Entrenamiento de modelos
SVM_OVA.fit(training_matrix, y_train)
SVM_OVO.fit(training_matrix, y_train)

# Prueba
y_out = SVM_OVA.predict(testing_matrix)
F1_score_OVA = 100*f1_score(y_test, y_out, average = 'weighted')
c_OVA = confusion_matrix(y_test, y_out)


y_out = SVM_OVO.predict(testing_matrix)
F1_score_OVO = 100*f1_score(y_test, y_out, average = 'weighted')
c_OVO = confusion_matrix(y_test, y_out)

'''--------------------------------------------- Punto 2 ----------------------------------------'''

# Importar datos de cada clase
X = np.genfromtxt('data/Punto 2/letter-recognition.csv', delimiter=',', skip_header=1, dtype=str)

# Tomando como referencia el nombre (Danie Zambrano) se encuentran las etiquetas para clasificación binaria:
y = np.zeros(len(X)) # Inicializar el vector etiquetas

# Etiquetas para clasificación binaria
for i in range(len(X)):
    if X[i][0] in 'DANIELZAMBRANO':
        y[i] = 0 # Letras presentes en los nombres
    else:
        y[i] = 1 # Letras no presentes en los nombres
X = np.float64(X[:,1:])

# Clases
presentes = X[np.where(y == 0)[0],:]
no_presentes = X[np.where(y == 1)[0],:]

# NOTA: CLASES NO HOMOGENEAS PRESENTES: 8464, NO PRESENTES: 11536
no_presentes = no_presentes[:len(presentes),:] # Balance de clases

# Segmentación de datos de datos en entrenamiento (80%) y prueba (20%)
Training_0, Testing_0 = model_selection.train_test_split(presentes,test_size = int(0.2*len(presentes)),train_size = int(0.8*len(presentes)))
Training_1, Testing_1 = model_selection.train_test_split(no_presentes,test_size = int(0.2*len(no_presentes)),train_size = int(0.8*len(no_presentes)))

# Creación de matrices de entrenamiento y prueba
training_matrix = np.concatenate((Training_0, Training_1), axis=0)
testing_matrix = np.concatenate((Testing_0, Testing_1), axis=0)
y_train = np.concatenate((np.zeros(len(Training_0)), np.ones(len(Training_1))), axis=0)
y_test = np.concatenate((np.zeros(len(Testing_0)), np.ones(len(Testing_1))), axis=0)

'''----------------------------------------- MLP -----------------------------------------'''
# Número de atributos
d = np.size(training_matrix, axis=1)

# Validación cruzada para tener (Validación (20% del total de datos o 25 % del conjunto de entrenamiento))
cv = KFold(n_splits=4, random_state=1, shuffle=True)

# Arquitecturas de MLP
# 1.) 16-8-1
mlp_1 = Sequential()
mlp_1.add(Dense(d, activation='tanh', input_shape=(d,)))
mlp_1.add(Dense(8, activation='tanh'))
mlp_1.add(Dense(1, activation='tanh'))

# 2.) 16-16-1
mlp_2 = Sequential()
mlp_2.add(Dense(d, activation='tanh', input_shape=(d,)))
mlp_2.add(Dense(16, activation='tanh'))
mlp_2.add(Dense(1, activation='tanh'))

# 3.) 16-32-1
mlp_3 = Sequential()
mlp_3.add(Dense(d, activation='tanh', input_shape=(d,)))
mlp_3.add(Dense(32, activation='tanh'))
mlp_3.add(Dense(1, activation='tanh'))

# 4.) 16-8-4-1
mlp_4 = Sequential()
mlp_4.add(Dense(d, activation='tanh', input_shape=(d,)))
mlp_4.add(Dense(8, activation='tanh'))
mlp_4.add(Dense(4, activation='tanh'))
mlp_4.add(Dense(1, activation='tanh'))

# 5.) 16-16-8-1
mlp_5 = Sequential()
mlp_5.add(Dense(d, activation='tanh', input_shape=(d,)))
mlp_5.add(Dense(16, activation='tanh'))
mlp_5.add(Dense(8, activation='tanh'))
mlp_5.add(Dense(1, activation='tanh'))

# 6.) 16-24-16-1
mlp_6 = Sequential()
mlp_6.add(Dense(d, activation='tanh', input_shape=(d,)))
mlp_6.add(Dense(24, activation='tanh'))
mlp_6.add(Dense(16, activation='tanh'))
mlp_6.add(Dense(1, activation='tanh'))

# Configuración de hiperparámetros adicionales: 
mlp_1.compile(optimizer = 'adam', loss = 'mean_squared_error')
mlp_2.compile(optimizer = 'adam', loss = 'mean_squared_error')
mlp_3.compile(optimizer = 'adam', loss = 'mean_squared_error')
mlp_4.compile(optimizer = 'adam', loss = 'mean_squared_error')
mlp_5.compile(optimizer = 'adam', loss = 'mean_squared_error')
mlp_6.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Errores
err_mlp_1 = []
err_mlp_2 = []
err_mlp_3 = []
err_mlp_4 = []
err_mlp_5 = []
err_mlp_6 = []

for train_indices, test_indices in cv.split(training_matrix):
    # Entrenamiento
    history_mlp_1 = mlp_1.fit(training_matrix[train_indices], y_train[train_indices], epochs = 250, 
              verbose = 1, workers = 8, use_multiprocessing = True,
              validation_data = (training_matrix[test_indices], y_train[test_indices]))
    history_mlp_2 = mlp_2.fit(training_matrix[train_indices], y_train[train_indices], epochs = 250, 
              verbose = 1, workers = 8, use_multiprocessing = True,
              validation_data = (training_matrix[test_indices], y_train[test_indices]))
    history_mlp_3 = mlp_3.fit(training_matrix[train_indices], y_train[train_indices], epochs = 250, 
              verbose = 1, workers = 8, use_multiprocessing = True,
              validation_data = (training_matrix[test_indices], y_train[test_indices]))
    history_mlp_4 = mlp_4.fit(training_matrix[train_indices], y_train[train_indices], epochs = 250, 
              verbose = 1, workers = 8, use_multiprocessing = True,
              validation_data = (training_matrix[test_indices], y_train[test_indices]))
    history_mlp_5 = mlp_5.fit(training_matrix[train_indices], y_train[train_indices], epochs = 250, 
              verbose = 1, workers = 8, use_multiprocessing = True,
              validation_data = (training_matrix[test_indices], y_train[test_indices]))
    history_mlp_6 = mlp_6.fit(training_matrix[train_indices], y_train[train_indices], epochs = 250, 
              verbose = 1, workers = 8, use_multiprocessing = True,
              validation_data = (training_matrix[test_indices], y_train[test_indices]))
    
    # Errores por cada validación realizada
    err_mlp_1.append(history_mlp_1.history['val_loss'])
    err_mlp_2.append(history_mlp_2.history['val_loss'])
    err_mlp_3.append(history_mlp_3.history['val_loss'])
    err_mlp_4.append(history_mlp_4.history['val_loss'])
    err_mlp_5.append(history_mlp_5.history['val_loss'])
    err_mlp_6.append(history_mlp_6.history['val_loss'])

# Errores promedio encontrados
err_bar_mlp_1 = np.mean(np.asarray(err_mlp_1))
err_bar_mlp_2 = np.mean(np.asarray(err_mlp_2))
err_bar_mlp_3 = np.mean(np.asarray(err_mlp_3))
err_bar_mlp_4 = np.mean(np.asarray(err_mlp_4))
err_bar_mlp_5 = np.mean(np.asarray(err_mlp_5))
err_bar_mlp_6 = np.mean(np.asarray(err_mlp_6))

# El mejor modelo es el de mlp 6

# Prueba:
y_hat = mlp_6.predict(testing_matrix)
y_out = abs(y_hat.round())

# Métricas de desempeño: 
c_dnn = confusion_matrix(y_test, y_out)
acc_dnn = 100*(c_dnn[0,0] + c_dnn[1,1])/sum(sum(c_dnn))
err_dnn = 100 - acc_dnn
se_dnn = 100*c_dnn[0,0]/(c_dnn[0,0] + c_dnn[0,1])
sp_dnn = 100*c_dnn[1,1]/(c_dnn[1,1] + c_dnn[1,0])