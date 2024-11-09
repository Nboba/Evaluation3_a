import numpy as np


def loadData():
    data = np.genfromtxt('KDDTrain.txt', delimiter=',',dtype=str,encoding='utf-8')

    
    tipoTrafico = {
        # Clase 1: Normal (Tráfico legítimo):
        1: ["normal"],
        # Clase 2: DOS (Denegación de Servicio) (Ataques de Denegación de Servicio):
        2: ["neptune","teardrop","smurf","pod","back","land","apache2","processtable","mailbomb","udpstorm"],
        # Clase 3: Probe (Sondeo) (Intentos de sondeo o exploración de la red):
        3: ["ipsweep","portsweep","nmap","satan","saint","mscan"]
    }
    var42 = data[:, 41]
    new_var42 = np.zeros_like(var42)

    for valor, claseLista in tipoTrafico.items():
        new_var42[np.isin(var42, claseLista)] = valor

    Y = new_var42

    X = data[:,:41]

    carac2 = np.unique(X[:,1])
    carac3 = np.unique(X[:,2])
    carac4 = np.unique(X[:,3])

    mapC2 = {valor: indice for indice, valor in enumerate(carac2,start=1)}
    mapC3 = {valor: indice for indice, valor in enumerate(carac3,start=1)}
    mapC4 = {valor: indice for indice, valor in enumerate(carac4,start=1)}
    
    for fila in X:
        fila[1] = mapC2[fila[1]]
        fila[2] = mapC3[fila[2]]
        fila[3] = mapC4[fila[3]]
    
    tipo1=[]
    tipo2=[]
    tipo3=[]

    for i,tipo in enumerate(Y):
        if tipo == '1':
            tipo1.append(X[i])
        if tipo == '2':
            tipo2.append(X[i])
        if tipo == '3':
            tipo3.append(X[i])

    np.savetxt("Data.csv", X.astype(float), delimiter=',', fmt='%f')     
    np.savetxt("classe1.csv",np.array(tipo1).astype(float), delimiter=',', fmt='%f')
    np.savetxt("classe2.csv", np.array(tipo2).astype(float), delimiter=',', fmt='%f')
    np.savetxt("classe3.csv", np.array(tipo3).astype(float), delimiter=',', fmt='%f')


def nMuestras(M):
    dataT1 = np.genfromtxt('classe1.csv', delimiter=',',dtype=float,encoding='utf-8')
    dataT2 = np.genfromtxt('classe2.csv', delimiter=',',dtype=float,encoding='utf-8')
    dataT3 = np.genfromtxt('classe3.csv', delimiter=',',dtype=float,encoding='utf-8')

    tipo1=[]
    tipo2=[]
    tipo3=[]

    for i in range(int(M)):
        tipo1.append(np.random.randint(0, len(dataT1)))
        tipo2.append(np.random.randint(0, len(dataT2)))
        tipo3.append(np.random.randint(0, len(dataT3)))
        
    np.savetxt("idx_class1.csv",np.array(tipo1).astype(int), delimiter=',', fmt='%i')
    np.savetxt("idx_class2.csv", np.array(tipo2).astype(int), delimiter=',', fmt='%i')
    np.savetxt("idx_class3.csv", np.array(tipo3).astype(int), delimiter=',', fmt='%i')


def crearDataSet():
    dataT1 = np.genfromtxt('classe1.csv', delimiter=',',dtype=float,encoding='utf-8')
    dataT2 = np.genfromtxt('classe2.csv', delimiter=',',dtype=float,encoding='utf-8')
    dataT3 = np.genfromtxt('classe3.csv', delimiter=',',dtype=float,encoding='utf-8')

    indices1 = np.genfromtxt('idx_class1.csv', delimiter=',',dtype=int,encoding='utf-8')
    indices2 = np.genfromtxt('idx_class2.csv', delimiter=',',dtype=int,encoding='utf-8')
    indices3 = np.genfromtxt('idx_class3.csv', delimiter=',',dtype=int,encoding='utf-8')

    dataset=[]

    for i in indices1:
        muestra = np.concatenate((dataT1[i], [1]),axis=0)
        dataset.append(muestra)
    for i in indices2:
        muestra = np.concatenate((dataT2[i], [2]),axis=0)
        dataset.append(muestra)
    for i in indices3:
        muestra = np.concatenate((dataT3[i], [3]),axis=0)
        dataset.append(muestra)
    

    np.savetxt("DataClasss.csv",np.array(dataset).astype(float), delimiter=',', fmt='%f')
