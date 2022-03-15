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

# División de datos de entrenamiento (85%) y prueba (15%):
training_a, testing_a = model_selection.train_test_split(X_a,test_size = int(0.15*len(X_a)),train_size = int(0.85*len(X_a)))
training_b, testing_b = model_selection.train_test_split(X_b,test_size = int(0.15*len(X_b)),train_size = int(0.85*len(X_b)))
training_c, testing_c = model_selection.train_test_split(X_c,test_size = int(0.15*len(X_c)),train_size = int(0.85*len(X_c)))

testing_matrix_abc = np.concatenate((testing_a, testing_b, testing_c),axis = 0)
training_matrix_abc = np.concatenate((training_a[0].reshape(1,2), training_b[0].reshape(1,2), training_c[0].reshape(1,2)),axis=0)

for i in range(len(training_a)-1):
    training_matrix_abc = np.concatenate((training_matrix_abc, np.concatenate((training_a[i+1].reshape(1,2), training_b[i+1].reshape(1,2), training_c[i+1].reshape(1,2)),axis=0)), axis=0)


y_train_a =  np.array([1,-1,-1]*len(training_a)).reshape(len(training_matrix_abc),1) # vectores teóricos (etiquetas)
y_train_b =  np.array([-1,1,-1]*len(training_a)).reshape(len(training_matrix_abc),1) # vectores teóricos (etiquetas)
y_train_c =  np.array([-1,-1,1]*len(training_a)).reshape(len(training_matrix_abc),1) # vectores teóricos (etiquetas)

y_test_a =  np.array([1,-1,-1]*len(testing_a)).reshape(len(testing_a)*3,1) # vectores teóricos (etiquetas)
y_test_b =  np.array([-1,1,-1]*len(testing_a)).reshape(len(testing_a)*3,1) # vectores teóricos (etiquetas)
y_test_c =  np.array([-1,-1,1]*len(testing_a)).reshape(len(testing_a)*3,1) # vectores teóricos (etiquetas)

# Visualización datos de entrenamiento
plt.scatter(training_a[:,0], training_a[:,1], c='red', label='Clase a')
plt.scatter(training_b[:,0], training_b[:,1], c='blue', label='Clase b')
plt.scatter(training_c[:,0], training_c[:,1], c='green', label='Clase c')
plt.title('Conjunto de datos data_2D ')
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')
plt.legend()
plt.grid()

plt.close()

def segmentacion_datos(training_matrix, i, valid_len):
    train_1 = training_matrix[:(valid_len*i),:] # Conjunto de entramiento a la izquierda de conjunto de validación
    train_2 = training_matrix[valid_len*(i+1):,:] # Conjunto de entramiento a la derecha de conjunto de validación
    train = np.concatenate((train_1,train_2), axis=0) # Conjunto total de entrenamiento
    valid = training_matrix[(valid_len*i):(valid_len*(i+1)),:] # Conjunto de validación

    return train, valid
 
'''
Algoritmo mínimos cuadrados (LMS):
'''
def LMS(train_data, validation_data, y_train, y_test):
    # Entrenamiento: 
    training_matrix = np.concatenate((train_data,np.ones((len(train_data),1))),axis = 1) # concatenamos x0 = 1
    A = np.matmul(np.transpose(training_matrix),training_matrix) # obtenemos la matriz A = sum(Xi*Xi')
    b = np.matmul(np.transpose(training_matrix),y_train) # obtenemos el vector b = sum(Xi*yi)

    if np.linalg.det(A) != 0:
        W = np.matmul(np.linalg.inv(A),b) # obtenemos el vector W por medido de la inversa de la matriz A (LMS)
    else: # la matriz A es mal condicionada o no invertible
        eta = 0.01 # tasa de aprendizaje
        W = np.zeros((d + 1,1))
        epochs = 100
        for k in range(epochs): 
            idx = np.random.permutation(len(training_matrix)) # vector de elementos aleatorios
            for i in range(len(training_matrix)): 
                h = np.matmul(np.transpose(W),np.transpose(training_matrix[idx[i],:])) # proyección del dato Xi en el vector W
                e = h - y_train[idx[i]] # error
                W = W - eta*np.transpose(training_matrix[idx[i],:]).reshape(d + 1,1)*e # obtenemos el vector W moviéndonos en dirección opuesta al gradiente del ECM (LMS Generalizado)
    
    # Prueba: 
    testing_matrix = np.concatenate((validation_data,np.ones((len(validation_data),1))),axis = 1) # concatenamos x0 = 1
    y_out = np.sign(np.transpose(np.matmul(np.transpose(W),np.transpose(testing_matrix)))) # si el dato es de la clase A, entonces w'x > 0 y viceversa

    # Métricas de rendimiento: 
    c_lms = confusion_matrix(y_test, y_out)
    acc_lms = 100*(c_lms[0,0] + c_lms[1,1])/sum(sum(c_lms))
    err_lms = 100 - acc_lms
    se_lms = 100*c_lms[0,0]/(c_lms[0,0] + c_lms[0,1])
    sp_lms = 100*c_lms[1,1]/(c_lms[1,1] + c_lms[1,0])

    return [W, acc_lms, err_lms, se_lms, sp_lms]

'''
Algoritmo discriminiante Logistico
'''
def sigmoid(x):
    sig = 1/(1 + math.exp(-x))
    return sig

def DL(train_data, validation_data, y_train, y_test):
    # Entrenamiento:
    training_matrix = np.concatenate((train_data,np.ones((len(train_data),1))),axis = 1) # concatenamos x0 = 1
    eta = 0.01 # tasa de aprendizaje
    W = np.zeros((d + 1,1))
    epochs = 100

    for k in range(epochs): 
        idx = np.random.permutation(len(training_matrix)) # vector de elementos aleatorios
        for i in range(len(training_matrix)): 
            h = np.matmul(np.transpose(W),np.transpose(training_matrix[idx[i],:])) # proyección del dato Xi en el vector W
            p = y_train[idx[i]]*(sigmoid(y_train[idx[i]]*h))*training_matrix[idx[i],:]  
            W = np.transpose(np.transpose(W) - eta*p) # obtenemos el vector W moviéndonos en dirección opuesta al gradiente de la función de costo (DL)
         
    # Prueba: 
    testing_matrix = np.concatenate((validation_data,np.ones((len(validation_data),1))),axis = 1) # concatenamos x0 = 1
    y_out = -np.sign(np.transpose(np.matmul(np.transpose(W),np.transpose(testing_matrix)))) # si el dato es de la clase A, entonces w'x > 0 y viceversa

    # Métricas de rendimiento: 
    c_dl = confusion_matrix(y_test, y_out)
    acc_dl = 100*(c_dl[0,0] + c_dl[1,1])/sum(sum(c_dl))
    err_dl = 100 - acc_dl
    se_dl = 100*c_dl[0,0]/(c_dl[0,0] + c_dl[0,1])
    sp_dl = 100*c_dl[1,1]/(c_dl[1,1] + c_dl[1,0])

    return [W, acc_dl, err_dl, se_dl, sp_dl]

'''
Algoritmo discriminante Fisher
'''
def DF(train_data_a, train_data_b, validation_data, y_test):
    # Entrenamiento: 
    M_a = np.mean(train_data_a,axis = 0).reshape(1,2) # media aritmética de la clase A 
    M_b = np.mean(train_data_b,axis = 0).reshape(1,2) # media aritmética de la clase B
    M = M_a - M_b # media aritmética entre clases

    train_data_ab = np.concatenate((train_data_a, train_data_b), axis=0)

    sI = np.zeros((2,2)) # inicializamos la matriz de covarianza (K) intra-clase

    for i in range(int(len(train_data_a))):
        sI = sI + np.matmul(np.transpose(train_data_a[i,:] - M_a),train_data_a[i,:] - M_a) 
    for i in range(int(len(train_data_b))):
        sI = sI + np.matmul(np.transpose(train_data_b[i,:] - M_b),train_data_b[i,:] - M_b) # actualización de sI

    W = np.matmul(np.linalg.inv(sI),np.transpose(M)) # obtenemos el vector W a través de la matriz de covarianza intra-clasde (DF)

    # Cálculo de w0:
    Mt = np.mean(train_data_ab,axis = 0).reshape(2,1) # media aritmética del conjunto de entrenamiento
    w0 = -np.matmul(np.transpose(W),Mt) # obtenemos el escalar w0 a través de la media aritmética de los datos de entrenamiento y del vector W (DF)

    # Prueba: 
    y_out = np.sign(np.transpose(np.matmul(np.transpose(W),np.transpose(validation_data)) + w0)) # si el dato es de la clase A, entonces w'x > 0 y viceversa

    # Métricas de rendimiento: 
    c_df = confusion_matrix(y_test, y_out)
    acc_df = 100*(c_df[0,0] + c_df[1,1])/sum(sum(c_df))
    err_df = 100 - acc_df
    se_df = 100*c_df[0,0]/(c_df[0,0] + c_df[0,1])
    sp_df = 100*c_df[1,1]/(c_df[1,1] + c_df[1,0])

    W = np.concatenate((W.reshape(2,1),w0.reshape(1,1)), axis=0)

    return [W, acc_df, err_df, se_df, sp_df]

'''
Algoritmo Perceptron
'''
def perceptron(train_data, validation_data, y_train, y_test):
    # Entrenamiento:
    training_matrix = np.concatenate((train_data,np.ones((len(train_data),1))),axis = 1) # concatenamos x0 = 1 
    eta = 0.01 # tasa de aprendizaje
    W = np.zeros((d + 1,1))
    epochs = 100
    for k in range(epochs): 
        idx = np.random.permutation(len(training_matrix)) # vector de elementos aleatorios
        for i in range(len(training_matrix)):
            h = np.matmul(np.transpose(W),np.transpose(training_matrix[idx[i],:])) # proyección del dato Xi en el vector W
            if h*y_train[idx[i]] <= 0: 
                W = W + eta*np.transpose(training_matrix[idx[i],:]).reshape(d + 1,1)*y_train[idx[i]]   
            
    # Prueba: 
    testing_matrix = np.concatenate((validation_data,np.ones((len(validation_data),1))),axis = 1) # concatenamos x0 = 1
    y_out = np.sign(np.transpose(np.matmul(np.transpose(W),np.transpose(testing_matrix)))) # si el dato es de la clase A, entonces w'x > 0 y viceversa

    # Métricas de rendimiento: 
    c_p = confusion_matrix(y_test, y_out)
    acc_p = 100*(c_p[0,0] + c_p[1,1])/sum(sum(c_p))
    err_p = 100 - acc_p
    se_p = 100*c_p[0,0]/(c_p[0,0] + c_p[0,1])
    sp_p = 100*c_p[1,1]/(c_p[1,1] + c_p[1,0])
    return [W, acc_p, err_p, se_p, sp_p]


k = 10
valid_len = int(len(training_matrix_abc)/k)

resultados_LMS_a = []
resultados_LMS_b = []
resultados_LMS_c = []

resultados_DL_a = []
resultados_DL_b = []
resultados_DL_c = []

resultados_DF_a = []
resultados_DF_b = []
resultados_DF_c = []

resultados_P_a = []
resultados_P_b = []
resultados_P_c = []

for i in range(k):
    # Datos de validación y entrenamiento para hiperplanos
    train_abc, valid_abc = segmentacion_datos(training_matrix_abc, i, valid_len)
    train_a = training_a[(17*i):(17*(i+1)),:]
    train_b = training_b[(17*i):(17*(i+1)),:]
    train_c = training_c[(17*i):(17*(i+1)),:]

    # Salidas para validación y entrenamiento
    y_train_a_cross, y_valid_a_cross = segmentacion_datos(y_train_a, i, valid_len)
    y_train_b_cross, y_valid_b_cross = segmentacion_datos(y_train_b, i, valid_len)
    y_train_c_cross, y_valid_c_cross = segmentacion_datos(y_train_c, i, valid_len)

    # Resulatados LMS
    resultados_LMS_a.append(LMS(train_abc, valid_abc, y_train_a_cross, y_valid_a_cross))
    resultados_LMS_b.append(LMS(train_abc, valid_abc, y_train_b_cross, y_valid_b_cross))
    resultados_LMS_c.append(LMS(train_abc, valid_abc, y_train_c_cross, y_valid_c_cross))

    # Resulatados DL
    resultados_DL_a.append(DL(train_abc, valid_abc, y_train_a_cross, y_valid_a_cross))
    resultados_DL_b.append(DL(train_abc, valid_abc, y_train_b_cross, y_valid_b_cross))
    resultados_DL_c.append(DL(train_abc, valid_abc, y_train_c_cross, y_valid_c_cross))

    # Resulatados DF
    resultados_DF_a.append(DF(train_a, np.concatenate((train_b, train_c),axis=0), valid_abc, y_valid_a_cross))
    resultados_DF_b.append(DF(train_b, np.concatenate((train_a, train_c),axis=0), valid_abc, y_valid_b_cross))
    resultados_DF_c.append(DF(train_c, np.concatenate((train_a, train_b),axis=0), valid_abc, y_valid_c_cross))

    # Resulatados Perceptron
    resultados_P_a.append(perceptron(train_abc, valid_abc, y_train_a_cross, y_valid_a_cross))
    resultados_P_b.append(perceptron(train_abc, valid_abc, y_train_b_cross, y_valid_b_cross))
    resultados_P_c.append(perceptron(train_abc, valid_abc, y_train_c_cross, y_valid_c_cross))


# Accuracy de cada hiperplano:
# LMS
accs_LMS_a = [resultado[1] for resultado in resultados_LMS_a]
accs_LMS_b = [resultado[1] for resultado in resultados_LMS_b]
accs_LMS_c = [resultado[1] for resultado in resultados_LMS_c]

# DL
accs_DL_a = [resultado[1] for resultado in resultados_DL_a]
accs_DL_b = [resultado[1] for resultado in resultados_DL_b]
accs_DL_c = [resultado[1] for resultado in resultados_DL_c]

# DF
accs_DF_a = [resultado[1] for resultado in resultados_DF_a] 
accs_DF_b = [resultado[1] for resultado in resultados_DF_b]
accs_DF_c = [resultado[1] for resultado in resultados_DF_c]

# P
accs_P_a = [resultado[1] for resultado in resultados_P_a]
accs_P_b = [resultado[1] for resultado in resultados_P_b]
accs_P_c = [resultado[1] for resultado in resultados_P_c]

# Mejor hiperplano par cada clase:
# LMS
H_a_LMS = resultados_LMS_a[accs_LMS_a.index(max(accs_LMS_a))][0]
H_b_LMS = resultados_LMS_b[accs_LMS_b.index(max(accs_LMS_b))][0]
H_c_LMS = resultados_LMS_c[accs_LMS_c.index(max(accs_LMS_c))][0]

# DL
H_a_DL = resultados_DL_a[accs_DL_a.index(max(accs_DL_a))][0]
H_b_DL = resultados_DL_b[accs_DL_b.index(max(accs_DL_b))][0]
H_c_DL = resultados_DL_c[accs_DL_c.index(max(accs_DL_c))][0]

# DF
H_a_DF = resultados_DF_a[accs_DF_a.index(max(accs_DF_a))][0]
H_b_DF = resultados_DF_b[accs_DF_b.index(max(accs_DF_b))][0]
H_c_DF = resultados_DF_c[accs_DF_c.index(max(accs_DF_c))][0]

# P
H_a_P = resultados_P_a[accs_P_a.index(max(accs_P_a))][0]
H_b_P = resultados_P_b[accs_P_b.index(max(accs_P_b))][0]
H_c_P = resultados_P_c[accs_P_c.index(max(accs_P_c))][0]

testing_matrix_abc = np.concatenate((testing_matrix_abc,np.ones((len(testing_matrix_abc),1))),axis = 1) # concatenamos x0 = 1
y_test_abc = np.concatenate((np.zeros((len(testing_a),1)), np.ones((len(testing_a),1)), 2*np.ones((len(testing_a),1))), axis=0)

def MLP(H_a, H_b, H_c, testing_matrix, y_test):
    y_out = np.zeros((len(testing_matrix),1))
    for i in range(len(testing_matrix)):
        ya = sigmoid(np.dot(H_a.T, testing_matrix[i]))
        yb = sigmoid(np.dot(H_b.T, testing_matrix[i]))
        yc = sigmoid(np.dot(H_c.T, testing_matrix[i]))

        out = (ya,yb,yc)
        clase = out.index(max(out))

        if clase == 0:
            y_out[i] = 0
        if clase == 1:
            y_out[i] = 1
        if clase == 2:
            y_out[i] = 2

    # Métricas de rendimiento:
    c_mlp = confusion_matrix(y_test, y_out)
    fp = c_mlp.sum(axis=0) - np.diag(c_mlp)  
    fn = c_mlp.sum(axis=1) - np.diag(c_mlp)
    tp = np.diag(c_mlp)
    tn = sum(sum(c_mlp)) - (fp + fn + tp)

    acc_mlp = 100*(tp+tn)/(tp+fp+fn+tn)
    err_mlp = 100 - acc_mlp
    se_mlp = 100*tp/(tp+fn)
    sp_mlp = 100*tn/(tn+fp)

    return [acc_mlp, err_mlp, se_mlp, sp_mlp]

# Resultados con metricas de rendimiento para MLP utilizando distintos tipos de clasificadores lineales
resultados_MLP_LMS = MLP(H_a_LMS, H_b_LMS, H_c_LMS, testing_matrix_abc, y_test_abc)
resultados_MLP_DL = MLP(-H_a_DL, -H_b_DL, -H_c_DL, testing_matrix_abc, y_test_abc)
resultados_MLP_DF = MLP(H_a_DF, H_b_DF, H_c_DF, testing_matrix_abc, y_test_abc)
resultados_MLP_P = MLP(H_a_P, H_b_P, H_c_P, testing_matrix_abc, y_test_abc)