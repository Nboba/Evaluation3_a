import numpy as np
import pprint

def config():
    param = np.genfromtxt('config.csv', delimiter=',', dtype=None,encoding='utf-8')
    return param

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normData(X):
    epsilon = 1e-10
    muX = np.mean(X, axis=0)
    sigmaX = np.std(X, axis=0) 
    u = (X - muX) / (sigmaX + epsilon)
    return sigmoid(u)


def embbeddingData(X,param):
    m= int(param[0])
    N = len(X)
    tau = int(param[1])
    c= int(param[2])
    M = int(N - (m - 1) * tau)
    embeddings = [X[i:i + m * tau:tau] for i in range(M)]
    Y = np.round(embeddings * c + 0.5).astype(int) 
    print(Y)
    return Y



def getPatron(Y,param):
    m= int(param[0])  # Longitud de cada embedding
    c = param[2]  # Constante utilizada en el paso 3
    # Crear el vector de potencias de c
    powersOfC = np.array([c**i for i in range(m)])
    patrones = [1 + np.dot(np.transpose(embedding) - 1, powersOfC) for embedding in Y]
    return patrones

def getFrecuencia(patrones,param):
    m= int(param[0]) 
    c = int(param[2])
    maxPatrones= c**m
    frecuencias = np.zeros(maxPatrones, dtype=int)
    for k in patrones:
        k = int(round())  # Redondea y convierte a entero
        frecuencias[k - 1] += 1
    return frecuencias
    
def probDispercion(frecuencias,N,param):
    m= param[0]
    tau = param[1]

    prob = frecuencias/(N-(m-1)*tau)
    print(prob)
    return prob
