# Information Gain
from math import sqrt
import numpy      as np
from util import utility_ig    as ut

# Dispersion entropy
def entropy_disp(data,param,N): 
    data = norm_data_sigmoidal(data)
    Y=ut.embbeddingData(data,param)
    patrones=ut.getPatron(Y,param)
    frecuencias=ut.getFrecuencia(patrones,N,param)
    DeNorm=ut.EntropyDispe(frecuencias,param)       
    return DeNorm

def norm_data_sigmoidal(data):
    dataNorm = ut.normData(data)
    return  dataNorm
#Information gain
def inform_gain():
    param=ut.config()
    data = load_data()
    X = data[:, :-1]  # Todas las columnas excepto la última
    Y = data[:, -1]
    unique_labels, Y_encoded = np.unique(Y, return_inverse=True)
    N, d = data.shape  
    B= int(sqrt(N)) 
    dataEntropy = entropy_disp(X,param,N)
    IG_values = {}
    for feature_idx in range(d):
        x = data[:, feature_idx]

        # Manejar valores NaN o infinitos
        x = np.nan_to_num(x)

        # Discretizar x en B bins
        x_min = np.min(x)
        x_max = np.max(x)
        bin_edges = np.linspace(x_min, x_max, B + 1)
        x_bins = np.digitize(x, bins=bin_edges, right=False)

        Hy_given_x = 0
        for b in np.unique(x_bins):
            indices = np.where(x_bins == b)[0]
            N_j = len(indices)
            if N_j == 0:
                continue
            Y_j = Y_encoded[indices]
            largo= len(Y_j)
            DE_Y_j = entropy_disp(Y_j,param,largo)
            w_j = N_j / N
            Hy_given_x += w_j * DE_Y_j

        IG = dataEntropy - Hy_given_x
        IG_values[feature_idx] = IG

    sorted_features = sorted(IG_values.items(), key=lambda item: item[1], reverse=True)

    # Mostrar las variables ordenadas
    print("Variables ordenadas por Ganancia de Información:")
    for feature_idx, IG in sorted_features:
        print(f"Característica {feature_idx}, Ganancia de Información: {IG}")
     

# Load dataClass 
def load_data():   
    return np.genfromtxt('DataClasss.csv', delimiter=',',dtype=float,encoding='utf-8')
# Beginning ...
def main(): 
    inform_gain() 
    pass
    
       
if __name__ == '__main__':   
	 main()

