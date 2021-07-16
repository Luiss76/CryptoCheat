import numpy as np
from Neural_Model import Neural_Model
from Data_Proc import Data_Set_Proc


'''
get the processed data
'''
data_set_proc = Data_Set_Proc()
data_set = data_set_proc.get_data_set('C:/Users/richa/Desktop/crypto-trading-ai-bot-basic-master/raw_historic_data/2021_01__BTC-USD.json')
data_set = np.append(data_set, [-1],axis = 0)

'''
generate neural network and feed forward to get output
'''
neural_model = Neural_Model([186,10,10,1], -1)
a = neural_model.forward(data_set)
print(a)