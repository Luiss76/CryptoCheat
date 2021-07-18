import numpy as np
from Neural_Model import Neural_Model
from Data_Proc import Data_Set_Proc
from Differential_Opt import Differential_Model
from time import time, ctime

'''
get the processed data
'''
data_set_proc = Data_Set_Proc()
data_set = data_set_proc.get_data_set('C:/Users/richa/Desktop/crypto-trading-ai-bot-basic-master/raw_historic_data/2021_01__BTC-USD.json')
data_set = np.append(data_set, [-1],axis = 0)
data_set_size = len(data_set)
#data_set_proc.plot_normalize_unormalize()
#print("time array :", data_set_proc.get_time_date())
'''
generate neural network and feed forward to get output
'''
neural_model = Neural_Model([data_set_size-1,2,2,1], -1)
print("\noutput of a single neural network :", neural_model.forward(data_set), "\n")


'''
init differential model and generate a population of neural network
'''

#set the desired population size
population_size = 5

#init differential model and generate population
differential_model = Differential_Model()
differential_model.gen_popu(population_size,neural_model)

#feed forward the input data set and fill in the output array of each neural network
output_array = np.empty(len(differential_model.popu_array), dtype=object)
for i in range(len(differential_model.popu_array)):
	output_array[i] = differential_model.popu_array[i].forward(data_set)
print("\noutput of  '", population_size,"' randomly generated population neural networks : \n\n", output_array)


#testing mutation function
differential_model.rand_1_mutation(0.5)
differential_model.current_to_best_1_mutation(differential_model.popu_array[0].network_topology,0.5,0.5)
#differential_model.test_current_to_best_1_mutation(differential_model.popu_array[0].network_topology,1.0,1.0)





#b = ctime(data_set[0])
#print(b)