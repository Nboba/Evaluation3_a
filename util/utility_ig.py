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
    X= X.flatten()
    N = len(X)
    m= int(param[0])
    tau = int(param[1])
    c= int(param[2])
    M = int(N - (m - 1) * tau)

    #Se crea el embbeding
    embeddings = np.array([X[i:i + m * tau:tau] for i in range(M)])
    #Se mapea el embbeding
    embeddingMap = np.round(embeddings * c + 0.5).astype(int) 
    embeddingMap[embeddingMap < 1] = 1
    embeddingMap[embeddingMap > c] = c
    return embeddingMap



def getPatron(data,param):
    m= int(param[0])  # Longitud de cada embedding
    c = param[2]  # Constante utilizada en el paso 3
    # Crear el vector de potencias de c
    cElevado_M = c ** np.arange(m)
    patrones = 1 + np.dot( data- 1, cElevado_M)
    return patrones

def getFrecuencia(patrones,N, param):
    patronesUnicos, frecuencia = np.unique(patrones, return_counts=True)
    probabilidades = frecuencia / frecuencia.sum()
    return probabilidades
    
def EntropyDispe(probabilidades,param):
    DE = -np.sum(probabilidades * np.log2(probabilidades))
    m= int(param[0])  # Longitud de cada embedding
    c = param[2]
    DENorm = DE / np.log2(c**m)
    return DENorm


