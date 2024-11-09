#-------------------------------------------------------
# Load Data from File: KDDTrain.txt
#--------------------------------------------------------

import numpy   as np
from util import utility_etl  as ut

# Load parameters from config.csv
def config():
    param = np.genfromtxt('config.csv', delimiter=',', dtype=None,encoding='utf-8')
    return param
# Beginning ...
def main():
    param=config() 
    ut.loadData()
    ut.nMuestras(param[0])
    ut.crearDataSet()
   
      
if __name__ == '__main__':   
	 main()

