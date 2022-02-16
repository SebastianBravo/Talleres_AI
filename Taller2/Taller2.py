import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection 

# Importar la informacion del archivo data
data = np.load('data.npy', allow_pickle=True)
data = data.item()

'''---------------------------- Punto 1 ----------------------------'''

# Separar los conjuntos de datos data_2D y data_3D
data_2D = data['data_2D']
data_3D = data['data_3D']

# Adquirir informacion de las 2 clases del conjunto de datos data_3D
data_3D_a = data_3D['data_a']
data_3D_b = data_3D['data_b']

# Division de datos en conjuntos de prueba y entrenamiento
training_a, testing_a = model_selection.train_test_split(data_3D_a, test_size=int(0.2*len(data_3D_a)), train_size=int(0.8*len(data_3D_a)))
training_b, testing_b = model_selection.train_test_split(data_3D_b, test_size=int(0.2*len(data_3D_b)), train_size=int(0.8*len(data_3D_b)))


# Se remueve la media de cada dato del conjunto de entrenamiento:
training_a_bar = training_a - np.mean(training_a, axis=0)
training_b_bar = training_b - np.mean(training_b, axis=0)

# Se calcula la matriz de covarianza de los datos
A_a = np.transpose(training_a_bar)
K_a = np.cov(A_a)

A_b = np.transpose(training_b_bar)
K_b = np.cov(A_b)

# Se encuentran los vectores y valores propios de K
eig_vals_a, eig_vecs_a = np.linalg.eig(K_a)
eig_vals_b, eig_vecs_b = np.linalg.eig(K_b)

# Se conservan solo los m valores propios más grandes y sus respectivos vectores propios
# Reduccion de R3 a R2 ---> m = 2
m = 2

# Para clase a:
M_a_pri = eig_vecs_a[:,:m] # Matriz de vectores propios reducida (dxm)
A_a_pri = np.matmul(np.transpose(M_a_pri), A_a) # Matriz de transformacion con los minimos componentes
new_training_a = np.transpose(A_a_pri) # Nueva matriz de caracteristicas

# Para clase a:
M_b_pri = eig_vecs_b[:,:m] # Matriz de vectores propios reducida (dxm)
A_b_pri = np.matmul(np.transpose(M_b_pri), A_b) # Matriz de transformacion con los minimos componentes
new_training_b = np.transpose(A_b_pri) # Nueva matriz de caracteristicas


# Visualizacion conjunto de entrenamiento
plt.figure(dpi = 150)
plt.scatter(new_training_a[:,0], new_training_a[:,1], c='red', label='Clase a')
plt.scatter(new_training_b[:,0], new_training_b[:,1], c='blue', label='Clase b')
plt.title('Conjunto de entrenamiento aplicando PCA')
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')
plt.legend()
plt.grid()


'''-----------------Clasificador Bayesiano Gaussiano-----------------'''
# Numero de atributos
n = m

# Probabilidades a pirori
P_3D_a = len(new_training_a)/(len(new_training_a) + len(new_training_b))
P_3D_b = len(new_training_b)/(len(new_training_a) + len(new_training_b))

# Calculo del centro de cada clase
u_3D_a = new_training_a.mean(axis=0)
u_3D_b = new_training_b.mean(axis=0)

# Matriz de covarianza:
K_3D_a = np.cov(np.transpose(new_training_a))
K_3D_b = np.cov(np.transpose(new_training_b))

# Matriz de prueba aplicando PCA

# Para clase a:
testing_a_bar = testing_a - np.mean(training_a, axis=0)
A_a_test_pri = np.matmul(np.transpose(M_a_pri), np.transpose(testing_a_bar)) 
new_testing_a = np.transpose(A_a_test_pri)

# Para clase b:
testing_b_bar = testing_b - np.mean(training_b, axis=0)
A_b_test_pri = np.matmul(np.transpose(M_b_pri), np.transpose(testing_b_bar)) 
new_testing_b = np.transpose(A_b_test_pri)

X = np.concatenate((new_testing_a, new_testing_b), axis=0)

# vector de comparacion
y_real = np.concatenate((np.zeros((len(new_testing_a),1)), np.ones((len(new_testing_b),1))), axis=0)

# vector con las clasificaciones
y_bayes = np.zeros((len(X),1))

for i in range(len(X)):
	# Funcion de verosimilitud de cada clase:
	P_a_X = P_3D_a*(1/(np.sqrt(2*(np.pi**n)*np.linalg.det(K_3D_a))))*(
			np.exp(-0.5*np.matmul(np.matmul(X[i,:]-u_3D_a,np.linalg.inv(K_3D_a)),X[i,:]-u_3D_a)))

	P_b_X = P_3D_b*(1/(np.sqrt(2*(np.pi**n)*np.linalg.det(K_3D_b))))*(
			np.exp(-0.5*np.matmul(np.matmul(X[i,:]-u_3D_b,np.linalg.inv(K_3D_b)),X[i,:]-u_3D_b)))

	# Vector de probabilidades
	P = (P_a_X, P_b_X)

	# Proceso de clasificacion 
	clase = P.index(max(P))

	# Clase a
	if clase==0:
		y_bayes[i] = 0

	# Clase b
	if clase==1:
		y_bayes[i] = 1

# Clasificacion del conjunto de prueba
a_bayes = np.array([X[i] for i in range(len(y_bayes)) if y_bayes[i] == 0])
b_bayes = np.array([X[i] for i in range(len(y_bayes)) if y_bayes[i] == 1])

# Visualizacion conjunto de prueba clasificado
plt.figure(dpi = 150)
plt.scatter(a_bayes[:,0], a_bayes[:,1], c='red', label='Clase a')
plt.scatter(b_bayes[:,0], b_bayes[:,1], c='blue', label='Clase b')
plt.title('Clasificacion realizada')
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')
plt.legend()
plt.grid()
plt.show()

# Carlculo error porcentual de la clasificacion
error_bayes = 100*sum(y_bayes != y_real)/len(y_real)
print(f"Error Bayes Gaussiano: {error_bayes[0]:.3f}%")