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
import time as tme



def evaluate_parent(best_parent_vec,parent_vector,training_data,min_parent_train_score):
	min_parent_score = 0
	valid_neural_score = 0
	initial_parent_score = best_parent_vec.score
	for i in range(len(parent_vector)):

		#LOOP THROUGH TRAINING DATA
		for j in range(len(training_data)):

			n_output = data_set_proc.unormalise(parent_vector[i].forward(data_set_proc.normalise(training_data[j],array_size)))
			min_parent_score = min_parent_score + parent_vector[i].error_funct(n_output,trn_ans_data[j])

			
		parent_vector[i].score = min_parent_score
		min_parent_score = 0

	return parent_vector

def evaluate_parent_V2(parent_vector,training_data):
	min_parent_score = 0
	valid_neural_score = 0
	for i in range(len(parent_vector)):

		#LOOP THROUGH TRAINING DATA
		for j in range(len(training_data)):

			n_output = data_set_proc.unormalise(parent_vector[i].forward(data_set_proc.normalise(training_data[j],array_size)))
			min_parent_score = min_parent_score + parent_vector[i].error_funct(n_output,trn_ans_data[j])
		parent_vector[i].score = min_parent_score
		min_parent_score = 0

	return parent_vector


'''
get the processed data
'''
array_size = 120
interval = 12
data_set_proc = Data_Set_Proc()
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


'''
generate neural network and feed forward to get output
'''

neural_model = Neural_Model([data_set_size-1,30,20,1], -1)
neural_weights = np.load("weights_6.npy",allow_pickle = True)

#neural_second_weights = np.load('sandbox_weight.npy',allow_pickle = True)
shape_arr =[len(neural_weights[0])]
for i in range(1,len(neural_weights)):
	shape_arr.append(len(neural_weights[i])-1)

import_neural = Neural_Model([shape_arr],-1)
import_second_neural = Neural_Model([shape_arr],-1)
import_neural.network_topology = neural_weights


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
population_size = 12

#set the crossover factor
crossover_factor = 0.9

#set the mutation factor 1 and 2
mutation_factor_1 = 0.1
mutation_factor_2 = 1.0

#init differential model and generate population
differential_model = Differential_Model()
parent_vector = differential_model.gen_popu(population_size,neural_model)
parent_vector[0] = import_neural

#feed forward the input data set and fill in the output array of each neural network
output_array = np.empty(len(differential_model.popu_array), dtype=object)


"""
calculate the first index of the popu to be set as the best vector and min error

"""




p = 1
best_parent_vec = []
cond = True
index = 0

best_parent_vec = parent_vector[1]
second_best_parent_vec = parent_vector[1]
min_parent_train_score = 0

min_parent_error = 0






#alid_score = 0
#valid_output = np.empty((len(testing_data),1)) 
#valid_output_graph = list()
#valid_ans_data_graph = list()


if __name__ == '__main__':
	try:
		for m in range(len(training_data)):
	
			min_parent_output = data_set_proc.unormalise(parent_vector[1].forward(data_set_proc.normalise(training_data[m],array_size)))
			min_parent_train_score = min_parent_train_score + parent_vector[1].error_funct(min_parent_output,trn_ans_data[m])
			#print("train score :",min_parent_train_score )
			parent_vector[1].score = min_parent_train_score

		print("best vec error :",best_parent_vec.score)
		while(1):
			start = tme.perf_counter()
			print("generation :",p)


			"""

			calculate the rest of the popu and actually get the best vector with lowest error

			"""
			s_4 = tme.perf_counter()
			n = math.floor(len(parent_vector)/4)
			print("starting now....")

			with Pool() as pool:

				parent_vector = pool.starmap(evaluate_parent,[(best_parent_vec,parent_vector[0:n],training_data,min_parent_train_score),(best_parent_vec,parent_vector[n:n*2],training_data,min_parent_train_score),
					(best_parent_vec,parent_vector[n*2:n*3],training_data,min_parent_train_score),(best_parent_vec,parent_vector[n*3:n*4],training_data,min_parent_train_score)])
				parent_vector = np.array(parent_vector).ravel()
				pool.close()
				pool.join()
				#print("parent_vector :",parent_vector,"parent vector score :",parent_vector[0].score)
				for i in range(len(parent_vector)):
					if(parent_vector[i].score < min_parent_train_score):
						print("NEW BEST VECTOR")
						min_parent_train_score = parent_vector[i].score
						second_best_parent_vec = best_parent_vec
						best_parent_vec = parent_vector[i]
						index = i
				print("best parent vec score:",best_parent_vec.score)

			f_4 = tme.perf_counter()
			print(f'parent evaluation {round(f_4-s_4, 2)} second(s)')
				

				#for l in range(len(parent_vector)):
			#		print("parent vector :",l," with score :",parent_vector[l].score)

			"""
			if(p % 1== 0):
				for i in range(50):
					out_n = best_parent_vec.forward(data_set_proc.normalise(training_data[i],array_size))
					best_vec_out = data_set_proc.unormalise(out_n)
					diff = best_parent_vec.get_diff(trn_ans_data[i])
					print("un normalised output :",out_n)
					print("min unormalise :",data_set_proc.unormalise(-1), "max unormalise :",data_set_proc.unormalise(1))
					a = [float(best_vec_out),float(best_vec_out)]
					b = [float(diff),float(diff)]
					plt.plot(x,a,'g')
					plt.plot(x,b,'y')
					plt.plot(trn_ans_data[i],'r')
					plt.pause(1.0)
					plt.clf()
			"""

			s_3 = tme.perf_counter()
			mutated_vector = differential_model.current_to_best_1_mutation(best_parent_vec.network_topology,mutation_factor_1,mutation_factor_2)
			

			#testing crossover function and its off spring
			offspring_vector = differential_model.binomial_crossover(parent_vector,mutated_vector,crossover_factor)
			f_3 = tme.perf_counter()
			print(f'mutation and crossover finishes in {round(f_3-s_3, 2)} second(s)')

			"""
			s_2 = tme.perf_counter()
			off_train_score = 0
			off_valid_score = 0
			for i in range(len(offspring_vector)):

				for z in range(len(training_data)):
					off_output = data_set_proc.unormalise(offspring_vector[i].forward(data_set_proc.normalise(training_data[z],array_size)))
					off_train_score = off_train_score + offspring_vector[i].error_funct(off_output,trn_ans_data[z])
				offspring_vector[i].score = off_train_score
				off_train_score = 0	
				#print("offspring vector score :",offspring_vector[i].score)
				#print("parent vector :",parent_vector[i].score)
				if(offspring_vector[i].score < parent_vector[i].score):
					
					print("\noff spring won")
					print("\n\noffspring training :",offspring_vector[i].score)
					print("parent training:",parent_vector[i].score)
					parent_vector[i] = offspring_vector[i]
						
			print("finish mutation")
			f_2 = tme.perf_counter()
			print(f'offspring evaluation in {round(f_2-s_2, 2)} second(s)')

			print("offspring training :",offspring_vector[0].score)
			"""
			start_timer = tme.perf_counter()

			n = math.floor(len(offspring_vector)/4)
			with Pool() as pool:
				offspring_vector_arr = pool.starmap(evaluate_parent_V2,[(offspring_vector[0:n],training_data),(offspring_vector[n:n*2],training_data),(offspring_vector[n*2:n*3],training_data),(offspring_vector[n*3:n*4],training_data)])
				offspring_vector = np.array(offspring_vector_arr).ravel()
				pool.close()
				pool.join()
			for i in range(len(offspring_vector)):

				#print("\n\noffspring training :",offspring_vector[i].score)
				#print("parent training:",parent_vector[i].score)

				if(offspring_vector[i].score < parent_vector[i].score):
					
					print("\noff spring won")
					print("\n\noffspring training :",offspring_vector[i].score)
					print("parent training:",parent_vector[i].score)
					parent_vector[i] = offspring_vector[i]

			#print("V2 offspring vector score :", offspring_vector[0][0].score)
			finish_timer = tme.perf_counter()
			

			print(f'new offspring method Finishes in {round(finish_timer-start_timer, 2)} second(s)')

			p = p+1
			finish = tme.perf_counter()
			print(f'Loop Finishes in {round(finish-start, 2)} second(s)')
	except KeyboardInterrupt:
		pass
		print('hello world')

	finally:	
		print("Finally")
		np.save( 'weights_8.npy' , best_parent_vec.network_topology )
		np.save('second_weight_8.npy',second_best_parent_vec.network_topology)
		