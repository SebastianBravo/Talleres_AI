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
Y_train = T_m[:,64]
T_m = T_m[:,:-1] # Remoción de etiquetas

# Creación matriz de prueba y etiquetas matriz de prueba.
Testing_matrix = np.concatenate((Testing_0,Testing_1,Testing_2,Testing_3), axis = 0)
Y_test =  Testing_matrix[:,64]
Testing_matrix = Testing_matrix[:,:-1] # Remoción de etiquetas

#Construccion de matriz de entrenamiento y validacion (Validación (20% del total de datos o 25 % del conjunto de entrenamiento))
train_0, val_0 = model_selection.train_test_split(Training_0, test_size= int(0.25*len(Training_0)), train_size = int(0.75*len(Training_0)))
train_1, val_1 = model_selection.train_test_split(Training_1, test_size= int(0.25*len(Training_1)), train_size = int(0.75*len(Training_1)))
train_2, val_2 = model_selection.train_test_split(Training_2, test_size= int(0.25*len(Training_2)), train_size = int(0.75*len(Training_2)))
train_3, val_3 = model_selection.train_test_split(Training_3, test_size= int(0.25*len(Training_3)), train_size = int(0.75*len(Training_3)))

Traning_matrix = np.concatenate((train_0,train_1,train_2,train_3), axis = 0)
Traning_matrix = Traning_matrix[:,:-1] # Remoción de etiquetas
Valid_matrix = np.concatenate((val_0,val_1,val_2,val_3), axis = 0)
Valid_matrix = Valid_matrix[:,:-1] # Remoción de etiquetas

# Etiquetas para entrenamiento y validacion
y_train = np.concatenate((np.zeros(len(train_0)), np.ones(len(train_1)), 2*np.ones(len(train_2)) , 3*np.ones(len(train_3))), axis = 0)
y_val = np.concatenate((np.zeros(len(val_0)), np.ones(len(val_1)), 2*np.ones(len(val_2)) , 3*np.ones(len(val_3))), axis = 0)

# Comprobación homogeneidad de datos
y_train_0 = len(np.where(Y_train == 0)[0]) 
y_train_1 = len(np.where(Y_train == 1)[0]) 
y_train_2 = len(np.where(Y_train == 2)[0]) 
y_train_3 = len(np.where(Y_train == 3)[0]) 

# NOTA: LOS DATOS NO SON HOMOGENEOS, PERO LA DIFERENCIA ENTRE EL NÚMERO DE DATOS ENTRE CLASES ES MENOR AL 1.5%

class_weights = {0:y_train_0/(y_train_1+y_train_2+y_train_3+y_train_0),
                1:y_train_1/(y_train_1+y_train_2+y_train_3+y_train_0),
                2:y_train_2/(y_train_1+y_train_2+y_train_3+y_train_0),
                3:y_train_3/(y_train_1+y_train_2+y_train_3+y_train_0),}

##SVM polinomial
###Construccion del modelo
SVM_poli_model = svm.SVC(C= 1.0,
                           gamma = 'auto',
                           degree = 3,
                           kernel = 'poly',
                           class_weight = class_weights,
                           decision_function_shape='ovr',
                           verbose=1)

SVM_poli_model.fit(Traning_matrix, y_train)
# Validación
y_out = SVM_poli_model.predict(Valid_matrix)
F1_score_SVM_poli = 100*f1_score(y_val, y_out, average = 'weighted')

# Prueba
y_out= SVM_poli_model.predict(Testing_matrix)
F1_score_SVM_poli = 100*f1_score(Y_test, y_out, average = 'weighted')