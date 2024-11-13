# Kernel-PCA by use Gaussian function

import numpy as np
from util import utility_kpca as ut

# Gaussian Kernel
def kernel_gauss(x, z, sigma):
    dist_sq = np.sum((x - z) ** 2)
    return np.exp(-dist_sq / (2 * sigma ** 2))

#Kernel-PCA
def kpca_gauss(X, sigma, top_k):

    #Calcular la matriz del kernel
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel_gauss(X[i], X[j], sigma)
    
    #Centrar la matriz del kernel
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    #Calcular los vectores y valores propios
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    
    #Ordenar los valores propios y los vectores propios
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Orden descendente
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    #Proyectar los datos sobre los primeros top_k componentes
    top_k_eigenvectors = eigenvectors_sorted[:, :int(top_k)]
    X_kpca = K_centered.dot(top_k_eigenvectors)
    
    return X_kpca

def load_data():
    return np.genfromtxt('DataIG.csv', delimiter=',')
    #return(x,y)                                                #esto estaba antes como return x,y pero nose pq retornaria eso

# Beginning ...
def main():			
    load_data()
    # Cargar los datos seleccionados por IG
    data = load_data()
    
    # Seleccionar las primeras 3000 muestras
    X = data[:3000]  
    # Guardar los datos seleccionados en un nuevo archivo Data.csv
    np.savetxt('Data.csv', X, delimiter=',', fmt='%f')

    param = ut.config()
    # Parámetros de KPCA
    sigma = param[4]  # Ejemplo de valor para el ancho del kernel
    top_k = param[5]   # Número de componentes principales a mantener
    
    # Aplicar KPCA
    X_kpca = kpca_gauss(X, sigma, top_k)
    
    # Guardar los datos proyectados en el nuevo archivo
    np.savetxt('DataKpca.csv', X_kpca, delimiter=',', fmt='%f')
		

if __name__ == '__main__':   
	 main()