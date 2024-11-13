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


def entropyDataSet(X,param):
    xT= X.transpose()
    shape=xT.shape

    m= int(param[0])
    tau = int(param[1])
    c= int(param[2])

    N = shape[1]
    M = int(N - (m - 1) * tau)

    embeddingsMap=[]
    # Se realiza el Embedding y el mapeo, Paso 2 y 3
    for carac in xT:
        embeddings=np.array([carac[i:i + m * tau:tau] for i in range(M)])
        embeddingMap = np.round(embeddings * c + 0.5).astype(int)
        embeddingMap[embeddingMap < 1] = 1
        embeddingMap[embeddingMap > c **m] = c **m
        embeddingsMap.append(embeddingMap)
    embeddingsMap = np.array(embeddingsMap)

    # Se obtienen los patrones paso 4 y 5
    cElevado_M = c ** np.arange(m)
    patrones = []
    for carac in embeddingsMap:
        patronesCarac = 1 + np.dot(carac - 1, cElevado_M)
        patrones.append(patronesCarac)
    patrones=np.array(patrones)
    # Se obtienen las frecuencias, Paso 5 y 6
    frecuenciaas=[]
    for pat in patrones:
        patronesUnicos, frecuencia = np.unique(pat, return_counts=True)
        probabilidades = frecuencia / frecuencia.sum()
        frecuenciaas.append(probabilidades)
    frecuenciaas=np.array(frecuenciaas)

    # Se calcula la entropía de dispersión y se normaliza Paso 7
    DENorm=0
    for prob in probabilidades:
        De = - np.sum(prob * np.log2(prob))
        DENorm += De / np.log2(c**m)


    return DENorm 

def entropyConditional(X,param):
    xT= X.transpose()
    largo=len(xT)
    m= int(param[0])
    tau = int(param[1])
    c= int(param[2])

    N = largo
    M = int(N - (m - 1) * tau)

    # Se realiza el Embedding y el mapeo, Paso 2 y 3
    embeddings = np.array([X[i:i + m * tau:tau] for i in range(M)])
    embeddingMap = np.round(embeddings * c + 0.5).astype(int) 
    embeddingMap[embeddingMap < 1] = 1
    embeddingMap[embeddingMap > c **m] = c **m

    # Se obtienen los patrones paso 4 y 5
    cElevado_M = c ** np.arange(m)
    patrones = 1 + np.dot( embeddingMap- 1, cElevado_M)

    # Se obtienen las frecuencias, Paso 5 y 6
    patronesUnicos, frecuencia = np.unique(patrones, return_counts=True)
    probabilidades = frecuencia / frecuencia.sum()

    # Se calcula la entropía de dispersión y se normaliza Paso 7
    De = -np.sum(probabilidades * np.log2(probabilidades))
    DENorm = De / np.log2(c**m)

    return DENorm
    
