import numpy as np # manejo de matrices
import matplotlib.pyplot as plt # gráficos
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Importar datos
dt = np.genfromtxt('basketball.dat', skip_header=8, delimiter=',')

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

#Descripcion Estadisitica.


# fig,axs= plt.subplots(5)
# axs[0].hist(dt[:,0],color="red",label="Asistencias")
# axs[1].hist(dt[:,1], color="blue", label= "Altura")
# axs[2].hist(dt[:,2], color="gray", label= "Tiempo jugado")
# axs[3].hist(dt[:,3], color="yellow", label= "Edad")
# axs[4].hist(dt[:,4], color="green", label= "Promedio de puntos jugado")

# axs[0].grid()
# axs[1].grid()
# axs[2].grid()
# axs[3].grid()
# axs[4].grid()

# fig.legend()

# plt.show()