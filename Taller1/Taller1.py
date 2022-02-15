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

# Calculo del centro de cada clase
u_2D_a = data_2D_a.mean(axis=1)
u_2D_b = data_2D_b.mean(axis=1)

# Calculo de las matrices de covarianza de cada clase
K_2D_a = np.cov(data_2D_a)
K_2D_b = np.cov(data_2D_b)

# Histograma clase_a data_2D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
hist, xedges, yedges = np.histogram2d(data_2D_a[0], data_2D_a[1], bins=(20,20))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.title('Histograma clase_a data_2D')
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')

# Histograma clase_b data_2D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
hist, xedges, yedges = np.histogram2d(data_2D_b[0], data_2D_b[1], bins=(20,20))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.title('Histograma clase_b data_2D')
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')

'''---------------------------- Punto 2 ----------------------------'''

# Adquirir informacion de las 2 clases del conjunto de datos data_3D
data_3D_a = data_3D['data_a']
data_3D_b = data_3D['data_b']

# Division de datos en conjuntos de prueba y entrenamiento
training_a, testing_a = model_selection.train_test_split(data_3D_a, test_size=int(0.2*len(data_3D_a)), train_size=int(0.8*len(data_3D_a)))
training_b, testing_b = model_selection.train_test_split(data_3D_b, test_size=int(0.2*len(data_3D_b)), train_size=int(0.8*len(data_3D_b)))

# Visualizacion de datos
plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter3D(training_a[:,0], training_a[:,1], training_a[:,2], c='red', label='Clase a')
ax.scatter3D(training_b[:,0], training_b[:,1], training_b[:,2], c='blue', label='Clase b')
plt.title('Conjunto de entrenamiento datos data_3D ')
ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.set_zlabel('Atributo 3')
plt.legend()
plt.grid()

'''-----------------Clasificador Bayesiano Gaussiano-----------------'''
# Numero de atributos
n = np.size(data_3D_a,axis=1)

# Probabilidades a pirori
P_3D_a = len(training_a)/(len(training_a) + len(training_b))
P_3D_b = len(training_b)/(len(training_a) + len(training_b))

# Calculo del centro de cada clase
u_3D_a = training_a.mean(axis=0)
u_3D_b = training_b.mean(axis=0)

# Matriz de covarianza:
K_3D_a = np.cov(np.transpose(training_a))
K_3D_b = np.cov(np.transpose(training_b))

# Matriz de prueba
X = np.concatenate((testing_a, testing_b), axis=0)

# vector de comparacion
y_real = np.concatenate((np.zeros((len(testing_a),1)), np.ones((len(testing_b),1))), axis=0)

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
plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter3D(a_bayes[:,0], a_bayes[:,1], a_bayes[:,2], c='red', label='Clase a')
ax.scatter3D(b_bayes[:,0], b_bayes[:,1], b_bayes[:,2], c='blue', label='Clase b')
plt.title('Conjunto de prueba datos data_3D clasificado con Bayes Gaussiano')
ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.set_zlabel('Atributo 3')
plt.legend()
plt.grid()

# Carlculo error porcentual de la clasificacion
error_bayes = 100*sum(y_bayes != y_real)/len(y_real)
print(f"Error Bayes Gaussiano: {error_bayes[0]:.3f}")

'''--------------Clasificador Bayesiano Gaussiano Naive---------------'''

# Probabilidades a pirori las misma

# Desviaciones estandar
std_a = training_a.std(axis=0)
std_b = training_b.std(axis=0)

# vector de comparacion
y_real = np.concatenate((np.zeros((len(testing_a),1)), np.ones((len(testing_b),1))), axis=0)

# vector con las clasificaciones
y_bayes_naive = np.zeros((len(X),1))

for i in range(len(X)):
	# Lista con probabilidades de atributos
	P_a_X = []
	P_b_X = []
	for j in range(n):
		# Calculo de probabilidades a posteriori de cada atributo
		P_a_X.append(P_3D_a*(1/(np.sqrt(2*(np.pi**n))*std_a[j]))*(np.exp(-0.5*(X[i,j]-u_3D_a)**2/std_a[j]**2)))
		P_b_X.append(P_3D_b*(1/(np.sqrt(2*(np.pi**n))*std_b[j]))*(np.exp(-0.5*(X[i,j]-u_3D_b)**2/std_b[j]**2)))

	# Probabilidad por clase
	P_a_X = np.prod(P_a_X)
	P_b_X = np.prod(P_b_X)

	# Vector de probabilidades
	P = (P_a_X, P_b_X)

	# Proceso de clasificacion 
	clase = P.index(max(P))

	# Clase a
	if clase==0:
		y_bayes_naive[i] = 0

	# Clase b
	if clase==1:
		y_bayes_naive[i] = 1

# Clasificacion del conjunto de prueba
a_bayes_naive = np.array([X[i] for i in range(len(y_bayes_naive)) if y_bayes_naive[i] == 0])
b_bayes_naive = np.array([X[i] for i in range(len(y_bayes_naive)) if y_bayes_naive[i] == 1])

# Visualizacion conjunto de prueba clasificado
plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter3D(a_bayes_naive[:,0], a_bayes_naive[:,1], a_bayes_naive[:,2], c='red', label='Clase a')
ax.scatter3D(b_bayes_naive[:,0], b_bayes_naive[:,1], b_bayes_naive[:,2], c='blue', label='Clase b')
plt.title('Conjunto de prueba datos data_3D clasificado con Bayes Gaussiano Naive')
ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.set_zlabel('Atributo 3')
plt.legend()
plt.grid()
plt.show()

# Carlculo error porcentual de la clasificacion
error_bayes_naive = 100*sum(y_bayes_naive != y_real)/len(y_real)
print(f"Error Bayes Gaussiano Naive: {error_bayes_naive[0]:.3f}")