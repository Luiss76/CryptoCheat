import numpy as np
import math
import json
import matplotlib.pyplot as plt

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

		'''
		Iterate through the neural topology and generate random neuron weight 

		'''

		for i in range(np.size(self.network_topology)):
			self.network_topology[i] = 0.1 * np.random.randn(n_neuron[i]+1,n_neuron[i+1])
		#return self.network_topology

	'''
	feed forward inputs to the network to get output

	'''
	def forward(self, inputs):
		max_range = np.size(self.network_topology)
		for i in range(max_range):
			if(i < 1):

				output = np.dot(inputs,self.network_topology[0]) 
				output = self.sigmoid_funct(output)
				output = np.append(output,[-1],axis = 0)
				#output = np.concatenate((output,-np.ones((np.shape(output)[0],1))),axis=1) 
				

			elif(i < max_range - 1):

				output = np.dot(output,self.network_topology[i])
				output = self.sigmoid_funct(output)
				output = np.append(output,[-1],axis = 0)

			else:
				output = np.dot(output,self.network_topology[i]) 
				output = self.sigmoid_funct(output)
				return output

		
	'''
	sigmoid activation function
	'''
	def sigmoid_funct(self,inputs):

		return 1.0/(1.0+np.exp(self.beta*inputs))


