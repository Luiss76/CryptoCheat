import numpy as np
import random as rnd
import copy
from Neural_Model import Neural_Model


class Differential_Model:

	def __init__(self):
		pass

	def gen_popu(self,popu_size,input_neural_model):

		'''
		generate population of neural networks

		'''
		
		self.popu_array = np.empty(popu_size, dtype = object)
		
		for i in range(popu_size):
			self.popu_array[i] = Neural_Model(input_neural_model.N_Neuron,input_neural_model.beta)
		

		return self.popu_array

	def rand_1_mutation(self,m_f):
		'''
		default rand/1 mutation function
		Vi,g = Xr1,g + F × (Xr2,g − Xr3,g)
		'''
		bull = True
		popu_size = np.size(self.popu_array)
		mutated_popu = np.empty(popu_size, dtype = object)
		for i in range(popu_size):
			while(bull):

				op1,op2,op3 = rnd.sample(range(0,popu_size), 3)
				if(op1 != i and op2 != i and op3 != i):
					bull = False
					m_op1 = np.array(self.popu_array[op1].network_topology)
					m_op2 = np.array(self.popu_array[op2].network_topology)
					m_op3 = np.array(self.popu_array[op3].network_topology)
					mutated_popu[i] = m_op1+m_f*(m_op2-m_op3)

			bull = True
		return mutated_popu


	def current_to_best_1_mutation(self,best_vec,m_f_1, m_f_2):
		'''
		DE/current to best/1 mutation scheme
		Vi,g = Xi,g + F1 × (Xbest,g − Xi,g) + F2 × (Xr1,g − Xr2,g)
		'''
		bull = True
		popu_size = np.size(self.popu_array)
		mutated_popu = copy.deepcopy(self.popu_array)	# deepcopy utilised to keep the neural attributes
		for i in range(popu_size):
			while(bull):
				op1 , op2 = rnd.sample(range(0,popu_size), 2)
				if(op1 != i and op2 != i):
					m_op1 = np.array(self.popu_array[op1].network_topology)
					m_op2 = np.array(self.popu_array[op2].network_topology)
					mutated_popu[i].network_topology = self.popu_array[i].network_topology + (m_f_1*(best_vec - self.popu_array[i].network_topology)) + (m_f_2*(m_op1 - m_op2))
					bull = False

			bull = True

		return mutated_popu

	def test_current_to_best_1_mutation(self,best_vec,m_f_1, m_f_2):
		'''
		TESTING DE/current to best/1 mutation scheme
		Vi,g = Xi,g + F1 × (Xbest,g − Xi,g) + F2 × (Xr1,g − Xr2,g)
		'''
		bull = True
		popu_size = np.size(self.popu_array)
		mutated_popu = np.empty(popu_size, dtype = object)
		for i in range(popu_size):
			print("i :", i)
			if(i == 0):
				print("the weight vector at index : ", i," is : \n", self.popu_array[i].network_topology[2])
				print("the best weight vector is : \n", best_vec[2])
			while(bull):
				op1 , op2 = rnd.sample(range(0,popu_size), 2)
				if(op1 != i and op2 != i):
					m_op1 = np.array(self.popu_array[op1].network_topology)
					m_op2 = np.array(self.popu_array[op2].network_topology)
					mutated_popu[i] = self.popu_array[i].network_topology + (m_f_1*(best_vec - self.popu_array[i].network_topology)) + (m_f_2*(m_op1 - m_op2))
					print("xbest,g - xi,g : " ,  (m_f_1*(best_vec - self.popu_array[i].network_topology))[2])
					print("m_op1 : \n", m_op1[2],"\nm_op2 : \n", m_op2[2])
					print("m_op1 - m_op2 : \n", m_op1[2] - m_op2[2])
					print("mutated popu : \n", mutated_popu[i][2])
					print("op1 :", op1, "op2 :", op2)
					bull = False
			bull = True

		return mutated_popu

	def binomial_crossover(self,parent_vec,mutated_vec,c_f):
		'''
		crossover function combining the parent and mutated vectors to produce offsprings
		'''
		vector_size = np.size(parent_vec)
		for i in range(vector_size):
			for j in range(np.size(parent_vec[i].network_topology)):
				for k in range(np.size(parent_vec[i].network_topology[j],axis = 0)):
					if (rnd.random() < c_f):
						parent_vec[i].network_topology[j][k] = mutated_vec[i].network_topology[j][k]
			
		return parent_vec
					
	def selection(self):
		'''
		selection
		'''
		pass

	def evaluate(self,output_array,score_array):
		
		'''
		evaluate	
		'''
		pass

	def run_evo(self):
		'''
		run evolution
		-how many times needed to be iterated or what target it needs to achieve
		-integrating with luis's code
		'''
		pass