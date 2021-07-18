import numpy as np
import json
import matplotlib.pyplot as plt
import statistics
import sys

				
				

			#time = np.append(time,["time"], axis = 0)
			#close_p = np.append(close_p, ["close_p"], axis = 0)
			#open_p = np.append(open_p, ["open_p"], axis = 0)

# [ time, low, high, open, close, volume ],
class Data_Set_Proc:

	def __init__(self):
		pass
	def get_data_set(self, address):


		with open(address,) as f:
			self.data = json.load(f)
			print(type(len(self.data)))
			self.data_set_size = len(self.data)
			self.time = np.empty((self.data_set_size,))
			low = np.empty((self.data_set_size,))
			high = np.empty((self.data_set_size,))
			open_p = np.empty((self.data_set_size,))
			self.close_p = np.empty((self.data_set_size,))
			volume = np.empty((self.data_set_size,))

			for z in range(len(self.data)):
				
				self.time[z] = self.data[z][0][0]
				low[z] = self.data[z][0][1]
				high[z] = self.data[z][0][2]
				open_p[z] = self.data[z][0][3]
				self.close_p[z] = self.data[z][0][4]
				volume[z] = self.data[z][0][5]

			#time = (time - statistics.mean(time))/statistics.stdev(time)
			low = (low - statistics.mean(low))/statistics.stdev(low)
			high = (high - statistics.mean(high))/statistics.stdev(high)
			open_p = (open_p - statistics.mean(open_p))/statistics.stdev(open_p)
			self.close_p = (self.close_p - statistics.mean(self.close_p))/statistics.stdev(self.close_p)
			volume = (volume - statistics.mean(volume))/statistics.stdev(volume)

			combined_data_set = (low, high, open_p, self.close_p,volume)

			combined_data_set =  np.hstack(combined_data_set)
			#print("COMBINE DATA SET  :", combined_data_set)

			return combined_data_set
	def get_time_date(self):
			return self.time

	def plot_normalize_unormalize(self):
		x = np.empty((self.data_set_size,))
		y = np.empty((self.data_set_size,))
		for i in range(self.data_set_size):
			x[i] = i
		for j in range(self.data_set_size):
			y[j] = self.data[j][0][4]

		figure, axis = plt.subplots(2, )
		axis[0].plot(x, y)
		axis[0].set_title("not normalized")

		axis[1].plot(x, self.close_p)
		axis[1].set_title("normalized")
		#plt.plot(x,y)
		#plt.plot(x,self.close_p)
		plt.show()