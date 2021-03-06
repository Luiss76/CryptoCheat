import numpy as np
from Neural_Model import Neural_Model
from Data_Proc import Data_Set_Proc


'''
get the processed data
'''
data_set_proc = Data_Set_Proc()
data_set = data_set_proc.get_data_set('C:/Users/richa/Desktop/Crypto_Cheat/CryptoCheat/raw_historic_data/2020_01__BTC-USD.json')
data_set = np.append(data_set, [-1],axis = 0)
data_set_size = len(data_set)

'''
generate neural network and feed forward to get output
'''
neural_model = Neural_Model([data_set_size-1,10,10,1], -1)
a = neural_model.forward(data_set)
print("Neural network output :", a)