import concurrent.futures
import time as tme
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
import matplotlib
from matplotlib import pyplot as plt
import math
from multiprocessing import Pool


def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    return f'Done Sleeping...{seconds}'



def loop_something(var):
	time_17,open_arr_17,close_arr_17,high_arr_17,low_arr_17,vol_arr_17 = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/V2/historical_data/1h_2017_1_to_2017_12_data.json')
	time_18,open_arr_18,close_arr_18,high_arr_18,low_arr_18,vol_arr_18 = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/V2/historical_data/1h_2018_1_to_2018_12_data.json')
	time_19,open_arr_19,close_arr_19,high_arr_19,low_arr_19,vol_arr_19 = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/V2/historical_data/1h_2019_1_to_2019_12_data.json')
	time_20,open_arr_20,close_arr_20,high_arr_20,low_arr_20,vol_arr_20 = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/V2/historical_data/1h_2020_1_to_2020_12_data.json')
	
	time =np.hstack((time_17,time_18,time_19,time_20))
	open_arr = np.hstack((open_arr_17,open_arr_18,open_arr_19,open_arr_20))
	close_arr=np.hstack((close_arr_17,close_arr_18,close_arr_19,close_arr_20))
	high_arr = np.hstack((high_arr_17,high_arr_18,high_arr_19,high_arr_20))
	low_arr = np.hstack((low_arr_17,low_arr_18,low_arr_19,low_arr_20))
	vol_arr = np.hstack((vol_arr_17,vol_arr_18,vol_arr_19,vol_arr_20))


	return close_arr


def evaluate_parent(best_parent_vec,parent_vector,training_data,min_parent_train_score):
		min_parent_score = 0
		valid_neural_score = 0
		for i in range(len(parent_vector)):

			#LOOP THROUGH TRAINING DATA
			for j in range(len(training_data)):

				n_output = data_set_proc.unormalise(parent_vector[i].forward(data_set_proc.normalise(training_data[j],array_size)))
				min_parent_score = min_parent_score + parent_vector[i].error_funct(n_output,trn_ans_data[j])

				
			parent_vector[i].score = min_parent_score
			#print("parent score :",parent_vector[i].score,"index :",i)
			min_parent_score = 0


			#print("previous best score :", min_parent_train_score)
			if(parent_vector[i].score < min_parent_train_score):
				#print("parent training :",parent_vector[i].score)
				#print("previous training:",min_parent_train_score)
				min_parent_train_score = parent_vector[i].score
				

				second_best_parent_vec = best_parent_vec
				best_parent_vec = parent_vector[i]
				index = i

		return best_parent_vec


array_size = 120
interval = 12
data_set_proc = Data_Set_Proc()
'''
get the processed data
'''

time_17,open_arr_17,close_arr_17,high_arr_17,low_arr_17,vol_arr_17 = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/V2/historical_data/1h_2017_1_to_2017_12_data.json')
time_18,open_arr_18,close_arr_18,high_arr_18,low_arr_18,vol_arr_18 = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/V2/historical_data/1h_2018_1_to_2018_12_data.json')
time_19,open_arr_19,close_arr_19,high_arr_19,low_arr_19,vol_arr_19 = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/V2/historical_data/1h_2019_1_to_2019_12_data.json')
time_20,open_arr_20,close_arr_20,high_arr_20,low_arr_20,vol_arr_20 = data_set_proc.get_import_data('C:/Users/richa/Desktop/Neural/V2/historical_data/1h_2020_1_to_2020_12_data.json')

time =np.hstack((time_17,time_18,time_19,time_20))
open_arr = np.hstack((open_arr_17,open_arr_18,open_arr_19,open_arr_20))
close_arr=np.hstack((close_arr_17,close_arr_18,close_arr_19,close_arr_20))
high_arr = np.hstack((high_arr_17,high_arr_18,high_arr_19,high_arr_20))
low_arr = np.hstack((low_arr_17,low_arr_18,low_arr_19,low_arr_20))
vol_arr = np.hstack((vol_arr_17,vol_arr_18,vol_arr_19,vol_arr_20))


data_set_proc.dissect_data(array_size,interval,open_arr,close_arr,low_arr,high_arr,vol_arr)

training_data = data_set_proc.unscaled_ds
trn_ans_data = data_set_proc.ans_data

#print("np sahpe of trn ans data :",np.shape(trn_ans_data[0]))

data_set_size = len(training_data[0])

x = [0,12]

neural_model = Neural_Model([data_set_size-1,30,20,1], -1)
neural_weights = np.load('weights_7.npy' ,allow_pickle = True)
shape_arr =[len(neural_weights[0])]
for i in range(1,len(neural_weights)):
	shape_arr.append(len(neural_weights[i])-1)

import_neural = Neural_Model([shape_arr],-1)
import_neural.network_topology = neural_weights

#set the desired population size
population_size = 16

#set the crossover factor
crossover_factor = 0.9

#set the mutation factor 1 and 2
mutation_factor_1 = 0.8
mutation_factor_2 = 0.8

#init differential model and generate population
differential_model = Differential_Model()
parent_vector = differential_model.gen_popu(population_size,neural_model)
parent_vector[0] = import_neural

#feed forward the input data set and fill in the output array of each neural network
output_array = np.empty(len(differential_model.popu_array), dtype=object)


if __name__ == '__main__':


	"""
	calculate the first index of the popu to be set as the best vector and min error

	"""

	p = 1
	best_parent_vec = []
	cond = True
	index = 0

	best_parent_vec = parent_vector[1]
	second_best_parent_vec = parent_vector[0]
	min_parent_train_score = 0

	min_parent_error = 0
	for m in range(len(training_data)):
		
		min_parent_output = data_set_proc.unormalise(parent_vector[1].forward(data_set_proc.normalise(training_data[m],array_size)))
		min_parent_train_score = min_parent_train_score + parent_vector[1].error_funct(min_parent_output,trn_ans_data[m])
	#print("train score :",min_parent_train_score )
	parent_vector[1].score = min_parent_train_score

	print("best vec error :",best_parent_vec.score)

	n = math.floor(len(parent_vector)/4)
	print("starting now....")
	start_5 = tme.perf_counter()
	with Pool() as pool:

		r_1 = pool.starmap(evaluate_parent,[(best_parent_vec,parent_vector[0:n],training_data,min_parent_train_score),(best_parent_vec,parent_vector[n:n*2],training_data,min_parent_train_score),
			(best_parent_vec,parent_vector[n*2:n*3],training_data,min_parent_train_score),(best_parent_vec,parent_vector[n*3:n*4],training_data,min_parent_train_score)])
		assert r_1
		#for i in range(len(r_1)):
		#	print("result of first :",r_1[i].score)
	finish_5 = tme.perf_counter()

	print(f'FIRST METHOD Finished in {round(finish_5-start_5, 2)} second(s)')

	start = tme.perf_counter()

	with concurrent.futures.ProcessPoolExecutor() as executor:
		n = math.floor(len(parent_vector)/4)
		
		results = executor.submit(evaluate_parent, best_parent_vec,parent_vector[0:n],training_data,min_parent_train_score)
		results_2 = executor.submit(evaluate_parent,best_parent_vec,parent_vector[n:n*2],training_data,min_parent_train_score)
		results_3 = executor.submit(evaluate_parent,best_parent_vec,parent_vector[n*2:n*3],training_data,min_parent_train_score)
		results_4 = executor.submit(evaluate_parent,best_parent_vec,parent_vector[n*3:n*4],training_data,min_parent_train_score)
		
		#print("\n\n first resut 1 :",results.result().score)
		#print("first resut 2 :",results_2.result().score)
		#print("first resutl 3:",results_3.result().score)
		#print("first resutl 4:",results_4.result().score)
		

	finish = tme.perf_counter()
	print(f'SECOND METHOD Finished in {round(finish-start, 2)} second(s)')

	start_2 = tme.perf_counter()

	best_vector = evaluate_parent(best_parent_vec,parent_vector,training_data,min_parent_train_score)

	finish_2 = tme.perf_counter()
	print(f'THIRD METHOD Finished in {round(finish_2-start_2, 2)} second(s)')


	start_3 = tme.perf_counter()
	min_parent_score = 0
	valid_neural_score = 0
	for i in range(len(parent_vector)):

		#LOOP THROUGH TRAINING DATA
		for j in range(len(training_data)):

			n_output = data_set_proc.unormalise(parent_vector[i].forward(data_set_proc.normalise(training_data[j],array_size)))
			min_parent_score = min_parent_score + parent_vector[i].error_funct(n_output,trn_ans_data[j])

			
		parent_vector[i].score = min_parent_score
		#print("parent score :",parent_vector[i].score,"index :",i)
		min_parent_score = 0


		#print("previous best score :", min_parent_train_score)
		if(parent_vector[i].score < min_parent_train_score):
			#print("parent training :",parent_vector[i].score)
			#print("previous training:",min_parent_train_score)
			min_parent_train_score = parent_vector[i].score
			

			second_best_parent_vec = best_parent_vec
			best_parent_vec = parent_vector[i]
			index = i

	finish_3 = tme.perf_counter()
	print(f'FOURTH METHOD Finished in {round(finish_3-start_3, 2)} second(s)')

		#print("best_parent_vec score :",best_parent_vec.score,"best vector score :",best_vector.score)

