# Information Gain

import numpy      as np
from util import utility_ig    as ut

# Dispersion entropy
def entropy_disp():   
    data = norm_data_sigmoidal()
    clases = data[:, len(data[0]) - 1]
    m, di = np.unique(clases, return_counts=True)

    pi = di/len(clases)         
    
    return -np.sum(pi*np.log2(pi))

def norm_data_sigmoidal():
    data = load_data()
    dataNorm = ut.sigmoid(data[:, :-1])
    dataNorm = np.hstack([dataNorm, data[:, -1].reshape(-1, 1)])
    return  dataNorm
#Information gain
def inform_gain():    
    pass

# Load dataClass 
def load_data():   
    return np.genfromtxt('DataClasss.csv', delimiter=',',dtype=float,encoding='utf-8')
# Beginning ...
def main():    
    pass
    
       
if __name__ == '__main__':   
	 main()

