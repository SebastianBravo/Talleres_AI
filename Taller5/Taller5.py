import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Importar la informacion del archivo data
data = np.load('data.npy', allow_pickle=True)
data = data.item()

# Division de datos en conjuntos de prueba y entrenamiento
training_matrix = data['training_set']
testing_matrix = data['testing_set']

'''------------------------- Punto 1 -------------------------'''
# Extracción datos variable dependiente e independiente

x = training_matrix[:,6][np.where(training_matrix[:,4] == 1)[0]] # Temperature para Holiday = 1
y = training_matrix[:,7][np.where(training_matrix[:,4] == 1)[0]] # Casual para Holiday = 1 *Debe ser número entero

x = x.reshape(len(x), 1)
y = y.reshape(len(y), 1)

pol = PolynomialFeatures(degree=3) # Grado del polinomio a usar
x_poly = pol.fit_transform(x) # Transformación de la entrada en polinomial
regression = LinearRegression() 
regression.fit(x_poly, y)

print(regression.score(x_poly,y))
print(regression.coef_, regression.intercept_)

y_pred = np.round(regression.predict(pol.fit_transform(testing_matrix.reshape(-1,1))))

# Visualizacion conjunto de entrenamiento
plt.scatter(x, y, c='red', label='Datos')
plt.plot(np.linspace(0,1,100), regression.predict(pol.fit_transform(np.linspace(0,1,100).reshape(-1,1))), c='blue', label='Regresión')
plt.title('Conjunto de entrenamiento')
plt.xlabel('Temperature')
plt.ylabel('Casual')
plt.xlim([0.1,1])
plt.ylim([0,4000])
plt.legend()
plt.grid()
plt.show()


