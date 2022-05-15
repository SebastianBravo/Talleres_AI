import numpy as np # manejo de matrices
import matplotlib.pyplot as plt # gráficos
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import random

# Importar datos
dt = np.genfromtxt('basketball.dat', skip_header=8, delimiter=',')

'''--------------------------Punto a-------------------------------'''

# Caracteristicas
assists = dt[:,0]
height = dt[:,1]
time = dt[:,2]
age = dt[:,3]
points = dt[:,4]

# Normalizacion de datos
def scaling(data):
	return (data-np.min(data))/(np.max(data)-np.min(data))

assists = scaling(assists)
height = scaling(height)
time = scaling(time)
age = scaling(age)
points = scaling(points)

# Histogrmas
fig,axs= plt.subplots(5)
axs[0].hist(assists,color="red",label="Asistencias")
axs[1].hist(height, color="blue", label= "Altura")
axs[2].hist(time, color="gray", label= "Tiempo jugado")
axs[3].hist(age, color="yellow", label= "Edad")
axs[4].hist(points, color="green", label= "Promedio de puntos jugado")

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()
axs[4].grid()
fig.legend()

# Descripcion Estadisitica [min, max, mean, var].
est_assists = [np.min(assists), np.max(assists), np.mean(assists), np.var(assists)]
est_height = [np.min(height), np.max(height), np.mean(height), np.var(height)]
est_time = [np.min(time), np.max(time), np.mean(time), np.var(time)]
est_age = [np.min(age), np.max(age), np.mean(age), np.var(age)]
est_points = [np.min(points), np.max(points), np.mean(points), np.var(points)]

# Diagrama de dispersión
plt.figure(2)
plt.boxplot([assists, height, time, age, points])
plt.xticks([1,2,3,4,5], ["Asistencias","Tiempo","Edad","Altura","Puntos"])

# Matriz de covarianza
matrix = np.concatenate((assists.reshape(-1,1), height.reshape(-1,1), time.reshape(-1,1), age.reshape(-1,1), points.reshape(-1,1)), axis = 1)
cov_matrix = np.cov(matrix.T)

'''--------------------------Punto b-------------------------------'''

# Matriz de distancias
z = linkage(matrix, "ward")

# Gráfica dendograma
plt.figure(3)
plt.title('Dendrograma')
plt.xlabel('Muestra')
plt.ylabel('Distancia')
plt.grid()
dendrogram(z, leaf_rotation=90)

# k means determine k
distortions = []
K = range(1,30)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(matrix)
    kmeanModel.fit(matrix)
    distortions.append(sum(np.min(cdist(matrix, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / matrix.T.shape[0])

# Plot the elbow
plt.figure(4)
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distorisión')
plt.title('Método de codo')
plt.grid()

# K means
k = 2 # 2 clases

# 1.) Definicion de hiperparametros:
epochs_kmeans = 500 # epocas del aprendizaje no supervisado

# 2.) Fase 1 (Aprendizaje no supervisado - k-means):
# centroides
c = np.zeros((k, np.size(matrix, axis=1))) # Inicializacion de la matriz de centroides
for i in range(k):
    for j in range(np.size(c, axis=1)):
        c[i,j] = random.uniform(min(matrix[:,j]), max(matrix[:,j])) # Inicializacion aleatoria

# Entrenamiento de k-means:
min_d = np.zeros(len(matrix)) # inicializacion del vector de minima distancia

print('Entrenamiento de k-means:\n')

for i in range(epochs_kmeans):
    print(f'Epoca: {i}')
    # Calcula de distancia a cada centroide:
    for j in range(len(matrix)):
        eu_d = np.sqrt(np.sum((matrix[j] - c)**2, axis=1)) # calculo de la distancia euclidiana
        min_d[j] = np.argmin(eu_d) # se escoge el centroide de menor distancia

    # Agrupacion de datos segun los centroides y actualizacion de los centroides
    for l in range(k):
        cluster = []
        for m in range(len(matrix)):
            if min_d[m] == l:
                cluster.append(matrix[m]) # agrupacion de datos del mismo cluster
            if len(cluster) == 1: # Si solo existe un dato, el centroide se desplaza a la posicion de este
                c[l] = np.asarray(cluster)
            elif len(cluster) > 1: # Si el cluster tiene varios datos, el centroide se actualiza de acuerdo a la media aritemtica
                c[l] = np.mean(np.asarray(cluster), axis=0) 

cluster1 = matrix[np.where(min_d==0)]
cluster2 = matrix[np.where(min_d==1)]

# Visualizacion de datos
plt.figure(5)
ax = plt.axes(projection ="3d")
ax.scatter3D(cluster1[:,1], cluster1[:,3], cluster1[:,4], c='red', label='Clase a')
ax.scatter3D(cluster2[:,1], cluster2[:,3], cluster2[:,4], c='blue', label='Clase b')
plt.title('Conjunto de entrenamiento datos data_3D ')
ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.set_zlabel('Atributo 3')
plt.legend()
plt.grid()

plt.show()