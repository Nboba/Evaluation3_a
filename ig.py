# Information Gain
from math import sqrt
import numpy      as np
from util import utility_ig    as ut

# Dispersion entropy
def entropy_disp(data,param,conditionalFlag): 
    data = norm_data_sigmoidal(data)
    if conditionalFlag == 0:  
        return ut.entropyDataSet(data,param) 
    else:
        return ut.entropyConditional(data,param)


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
    N, d = X.shape  
    B= int(sqrt(N)) 
    dataEntropy = entropy_disp(X,param,0)
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
            Y_j = x[indices]
            DE_Y_j = entropy_disp(Y_j,param,1)
            w_j = N_j / N
            Hy_given_x += w_j * DE_Y_j

        IG = dataEntropy - Hy_given_x
        IG_values[feature_idx] = IG

    sorted_features = sorted(IG_values.items(), key=lambda item: item[1], reverse=True)

    # Mostrar las variables ordenadas
    print("Variables ordenadas por Ganancia de Información:")
    indices = []
    for feature_idx, IG in sorted_features:
        print(f"Característica {feature_idx}, Ganancia de Información: {IG}")
        indices.append(feature_idx-1)
    topK=int(param[3])
    top_k_indices =  indices[0:topK]
    np.savetxt("Idx_variable.csv", top_k_indices, delimiter=',', fmt='%d')

    # Crear el conjunto de datos X_K con solo las Top-K variables
    top_k = X[:, top_k_indices]
    data_top_k = np.column_stack((top_k, Y))  # Combina con la columna Y
    np.savetxt("DataIG.csv", data_top_k, delimiter=',', fmt='%f')
     

# Load dataClass 
def load_data():   
    return np.genfromtxt('DataClasss.csv', delimiter=',',dtype=float,encoding='utf-8')
# Beginning ...
def main(): 
    inform_gain() 
    pass
    
       
if __name__ == '__main__':   
	 main()

