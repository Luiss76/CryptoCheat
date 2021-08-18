import numpy as np
from Neural_Model import Neural_Model
from Data_Proc import Data_Set_Proc
from Differential_Opt import Differential_Model
from time import time, ctime
import datetime as dt              # working with dates
import requests 
import json
from numpyencoder import NumpyEncoder
from sklearn.preprocessing import MinMaxScaler
'''
get the processed data
'''
array_size = 5
interval = 3
data_set_proc = Data_Set_Proc()
time,open_arr,close_arr,high_arr,low_arr,vol_arr = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/raw_historic_data/2021_1_to_2021_2_data.json')
data_set_proc.dissect_data(array_size,interval)
training_data, validation_data, testing_data = data_set_proc.produce_trest(75,25)
print("size of training_data :", np.shape(training_data))
data_set = training_data[0]
data_set_size = len(training_data[0])
#print("close_arr :", len(close_arr))

#data_set = np.append(data_set, [-1],axis = 0)
#data_set_time = data_set_proc.get_time()
#data_set_size = len(data_set)

'''
generate neural network and feed forward to get output
'''
neural_model = Neural_Model([data_set_size-1,2,10,1], -1)
print("\noutput of a single neural network :", neural_model.forward(data_set), "\n")
'''
DE PROCESS
-GENERATE INIT PARENT POPULATION
-EVALUATE THE PARENT POPULATION
-MUTATE THE PARENT POPULATION
-CROSSOVER THE MUTATED AND PARENT POPULATIONS
-SELECT WHETHER THE OFFSPRINGS ARE BETTER THAN THE PARENTS
-REPEAT 
'''

#set the desired population size
population_size = 5

#set the crossover factor
crossover_factor = 0.5

#set the mutation factor 1 and 2
mutation_factor_1 = 1.0
mutation_factor_2 = 1.0

#init differential model and generate population
differential_model = Differential_Model()
parent_vector = differential_model.gen_popu(population_size,neural_model)


#feed forward the input data set and fill in the output array of each neural network
output_array = np.empty(len(differential_model.popu_array), dtype=object)
print("size of training data :",np.shape(training_data))
print("size of answers of training data :", np.shape(data_set_proc.ans_training_data))
print("answer training data :", data_set_proc.ans_training_data[0])


for i in range(len(differential_model.popu_array)):
	output_array[i] = data_set_proc.scaler_close.inverse_transform(differential_model.popu_array[i].forward(data_set).reshape(-1,1))
	#print("training size  0:",training_data[0], "training_size 1:",training_data[1])
	differential_model.popu_array[i].score = output_array[i] - np.max(data_set_proc.ans_training_data[0])
#print("\n\nmax answer of training data :", np.max(data_set_proc.ans_training_data[0]))
#print("\noutput of  '", population_size,"' randomly generated population neural networks : \n\n", output_array)

for i in range(len(differential_model.popu_array)):
	print("predicted price of neural",i," :", output_array[i])
	print("actual price of next 15  :", data_set_proc.ans_training_data[0])
	print("score of population neural",i,":",differential_model.popu_array[i].score )
	print("\n")

#testing mutation function
rand_1_mutated_array = differential_model.rand_1_mutation(0.5)
mutated_vector = differential_model.current_to_best_1_mutation(differential_model.popu_array[0].network_topology,mutation_factor_1,mutation_factor_2)
#differential_model.test_current_to_best_1_mutation(differential_model.popu_array[0].network_topology,1.0,1.0)

#testing crossover function and its off spring
offspring_vector = differential_model.binomial_crossover(parent_vector,mutated_vector,crossover_factor)

for i in range(len(offspring_vector)):
	output_array[i] = offspring_vector[i].forward(data_set)
print("\noutput of '", population_size,"' crossover offspring population neural networks : \n\n", output_array)



#print(c[1].network_topology[2])

#b = ctime(data_set[0])
#print(b)