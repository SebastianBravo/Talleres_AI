import numpy as np # manejo de matrices
import matplotlib.pyplot as plt # gráficos
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

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