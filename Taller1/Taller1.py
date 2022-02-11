import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection #Segmentacion de datos

# Importar la informacion del archivo data
data = np.load('data.npy', allow_pickle=True)
data = data.item(0)


'''---------------------------- Punto 1 ----------------------------'''

# Separar los conjuntos de datos data_2D y data_3D
data_2D = data['data_2D']
data_3D = data['data_3D']

# Adquirir informacion de las 2 clases del conjunto de datos data_2D
data_2D_a = data_2D['data_a']
data_2D_b = data_2D['data_b']

# Grafica de los datos con colores distintivos para cada clase
plt.scatter(data_2D_a[0], data_2D_a[1], c='red', label='Clase a')
plt.scatter(data_2D_b[0], data_2D_b[1], c='blue', label='Clase b')
plt.title('Conjunto de datos data_2D ')
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')
plt.legend()
plt.grid()
#plt.show()

# Calculo del centro de cada clase
u_2D_a = data_2D_a.mean(axis=1)
u_2D_b = data_2D_b.mean(axis=1)

# Calculo de las matrices de covarianza de cada clase
K_2D_a = np.cov(data_2D_a)
K_2D_b = np.cov(data_2D_b)

'''---------------------------- Punto 2 ----------------------------'''

# Adquirir informacion de las 2 clases del conjunto de datos data_3D
data_3D_a = data_3D['data_a']
data_3D_b = data_3D['data_b']

# Division de datos en conjuntos de prueba y entrenamiento
training_a, testing_a = model_selection.train_test_split(data_3D_a, test_size=int(0.2*len(data_3D_a)), train_size=int(0.8*len(data_3D_a)))
training_b, testing_b = model_selection.train_test_split(data_3D_b, test_size=int(0.2*len(data_3D_b)), train_size=int(0.8*len(data_3D_b)))

# Visualizacion de datos

'''-----------------Clasificar Bayesiano Gaussiano-----------------'''
# Numero de atributos
n = np.size(data_3D_a,axis=1)

# Probabilidades a pirori
P_3D_a = len(training_a)/(len(training_a) + len(training_b))
P_3D_b = len(training_b)/(len(training_a) + len(training_b))

# Calculo del centro de cada clase
u_3D_a = training_a.mean(axis=0)
u_3D_b = training_a.mean(axis=0)

# Matriz de covarianza:
K_3D_a = np.cov(np.transpose(training_a))
K_3D_b = np.cov(np.transpose(training_b))

# Matriz de prueba
X = np.concatenate((testing_a, testing_b), axis=0)
y_real = np.concatenate((np.zeros((len(testing_a),1)), np.ones((len(testing_b),1))), axis=0)
y_bayes = np.zeros((len(X),1))

for i in range(len(X)):
	# Funcion de verosimilitud de cada clase:
	P_a_X = P_3D_a*(1/(np.sqrt(2*(np.pi**n)*np.linalg.det(K_3D_a))))*(np.exp(-0.5*np.matmul(np.matmul(X[i,:]-u_3D_a,np.linalg.inv(K_3D_a)),X[i,:]-u_3D_a)))
	P_b_X = P_3D_b*(1/(np.sqrt(2*(np.pi**n)*np.linalg.det(K_3D_b))))*(np.exp(-0.5*np.matmul(np.matmul(X[i,:]-u_3D_b,np.linalg.inv(K_3D_b)),X[i,:]-u_3D_b)))

	# Vector de probabilidades
	P = (P_a_X, P_b_X)

	# Proceso de clasificacion 
	clase = P.index(max(P))

	if clase==0:
		y_bayes[i] = 0
	if clase==1:
		y_bayes[i] = 1

# Carlculo error porcentual de la clasificacion
error_bayes = 100*sum(y_bayes != y_real)/len(y_real)
print(error_bayes)