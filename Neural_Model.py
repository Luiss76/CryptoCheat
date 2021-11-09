import numpy as np
import math
import json
import matplotlib.pyplot as plt
import statistics as st

class Neural_Model:
	'''
	Class that generates dynamic amount of neurons and hidden layers

	'''
	def __init__(self,n_neuron,beta):
		'''
		get required number of neurons and hidden layers

		n_neuron = [n_input_neuron, n_hidd_neuron, ..., n_output_neuron]

		'''
		self.N_Neuron = n_neuron
		#print("N_Neuron :", self.N_Neuron)
		self.beta = beta
		self.n_layer = len(n_neuron) - 1
		self.network_topology = np.empty(self.n_layer, dtype= object)
		self.score = 0
		self.valid_score = 0

		'''
		Iterate through the neural topology and generate random neuron weight 

		'''

		for i in range(np.size(self.network_topology)):
			self.network_topology[i] = 0.1 * np.random.randn(n_neuron[i]+1,n_neuron[i+1])
		#return self.network_topology

	'''
	feed forward inputs to the network to get output

	'''
	def import_model(self,neural_model,address):
		neural_weights = np.load(address,allow_pickle = True)
		shape_arr =[len(neural_weights[0])]
		for i in range(1,len(neural_weights)):
			shape_arr.append(len(neural_weights[i])-1)

		neural_model = Neural_Model([shape_arr],-1)
		neural_model.network_topology = neural_weights

	def forward(self, inputs):
	
		inputs = inputs.flatten()
		inputs = np.reshape(inputs,(1,len(inputs)))
		#print("shape of input :",np.shape(inputs))
		max_range = np.size(self.network_topology)
		for i in range(max_range):
			if(i < 1):

				output = np.dot(inputs,self.network_topology[0])
				output = self.sigmoid_funct(output)
				#print("output before appended :", output)
				output = np.concatenate((output,-np.ones((np.shape(output)[0],1))),axis=1)
				#output = np.concatenate((output,-np.ones((np.shape(output)[0],1))),axis=1)
				

			elif(i < max_range - 1):

				output = np.dot(output,self.network_topology[i])
				output = self.sigmoid_funct(output)
				output = np.concatenate((output,-np.ones((np.shape(output)[0],1))),axis=1)
			else:
				output = np.dot(output,self.network_topology[i]) 
				output = self.tanh(output)
				return output

	def error_funct(self,output,target):
		#print("targets :",np.shape(target))
		#print("targets :", target)
		#print("max value :",max(target))		

		upper_error = abs(max(target) - output)
		lower_error = abs(min(target) - output)

		mean_target = st.mean(target)
		median_target = st.median(target)
		diff = 0
		weight = 1.0
		w_arr = []

		time_weight_interval = 0.2/len(target)
		time_weight = time_weight_interval
		tw_arr = []


		for i in range(len(target)):
			w_arr = np.append(w_arr,abs((target[i] - median_target)))
			tw_arr = np.append(tw_arr,time_weight)
			time_weight = time_weight + time_weight_interval
			
		w_arr = w_arr/max(w_arr)
		#print("time weight :",tw_arr)
		
		

		w_arr = (abs((w_arr - st.mean(w_arr))/st.stdev(w_arr))) + 1
		
		w_arr = (w_arr / (1.5*max(w_arr)))+1
		#w_arr [np.argmax(w_arr)] = w_arr [np.argmax(w_arr)] * 0.5
		#print("w_arr :",w_arr)
		w_arr = w_arr + tw_arr
		#print("newly added w_arr:",w_arr)
		
		for i in range(len(target)):
			#print("diff added by :", (target[i] - mean_target)*w_arr[i])
			diff = diff + ((target[i] - mean_target)*w_arr[i])
			#print("diff :", diff)
			#diff = diff + ((target[i] - mean_target)*weight)
			#weight = weight + 0.05

		diff = diff + mean_target
		error = abs(output - diff)
		#if(upper_error > lower_error):
		#	error = lower_error
		#else:
		#		error = upper_error
		#print("diff:",diff)
		output = output.reshape(1,)
		#x = [0,12]
		#y = [output,output]
		#c = [median_target, median_target]
		#p = [mean_target, mean_target]
		#l = [diff , diff]
		#print("shape of output :", np.shape(output))
		#plt.plot(target,'r')
		#plt.plot(x,p,'g')
		#plt.plot(x,c,'b')
		#plt.plot(x,l,'y')
		#plt.show()

		return error

	def get_diff(self,target):
		mean_target = st.mean(target)
		median_target = st.median(target)
		diff = 0
		weight = 1.0
		w_arr = []

		time_weight_interval = 0.2/len(target)
		time_weight = time_weight_interval
		tw_arr = []


		for i in range(len(target)):
			w_arr = np.append(w_arr,abs((target[i] - median_target)))
			tw_arr = np.append(tw_arr,time_weight)
			time_weight = time_weight + time_weight_interval
			
		w_arr = w_arr/max(w_arr)
		#print("time weight :",tw_arr)
		
		

		w_arr = (abs((w_arr - st.mean(w_arr))/st.stdev(w_arr))) + 1
		
		w_arr = (w_arr / (1.5*max(w_arr)))+1
		#w_arr [np.argmax(w_arr)] = w_arr [np.argmax(w_arr)] * 0.5
		#print("w_arr :",w_arr)
		w_arr = w_arr + tw_arr
		#print("newly added w_arr:",w_arr)
		
		for i in range(len(target)):
			#print("diff added by :", (target[i] - mean_target)*w_arr[i])
			diff = diff + ((target[i] - mean_target)*w_arr[i])
			#print("diff :", diff)
			#diff = diff + ((target[i] - mean_target)*weight)
			#weight = weight + 0.05

		diff = diff + mean_target

		return diff

		
	'''
	sigmoid activation function
	'''
	def sigmoid_funct(self,inputs):
		inputs = np.clip(inputs, -501,501)
		return 1.0/(1.0+np.exp(self.beta*inputs))

	def tanh(self,inputs):
		inputs = np.clip(inputs,-501,501)
		y = (np.exp(inputs) - np.exp(-inputs))/(np.exp(inputs) + np.exp(-inputs))
		return y
