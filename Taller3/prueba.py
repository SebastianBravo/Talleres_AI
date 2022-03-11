"""
Desarrollado por:   Juan Sebastián Bravo Santacruz
                    Daniel Stiven Zambrano Acosta
                    Sergio Lavao Osorio
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

db = np.load('data_2D.npy',allow_pickle = True).item()

# Segmentación de datos: 
X_a = db['A'] # patrones de la clase A
X_b = db['B'] # patrones de la clase B
X_c = db['C'] # patrones de la clase B

X = np.concatenate((X_a,X_b,X_c),axis = 0)
d = np.size(X_a,axis = 1) # atributos de la clase A == atributos de la clase B == atributos de la clase C

# División de datos de entrenamiento (70%) y prueba (30%):
training_a, testing_a = model_selection.train_test_split(X_a,test_size = int(0.15*len(X_a)),train_size = int(0.85*len(X_a)))
training_b, testing_b = model_selection.train_test_split(X_b,test_size = int(0.15*len(X_b)),train_size = int(0.85*len(X_b)))
training_c, testing_c = model_selection.train_test_split(X_c,test_size = int(0.15*len(X_c)),train_size = int(0.85*len(X_c)))

a = np.array([1,-1,-1]*10)