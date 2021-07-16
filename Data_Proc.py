import numpy as np
import json
import matplotlib.pyplot as plt
import statistics
import sys


# [ time, low, high, open, close, volume ],
class Data_Set_Proc:

	def __init__(self):
		pass
	def get_data_set(self, address):


		with open(address,) as f:
			data = json.load(f)
			print(type(len(data)))
			data_set_size = len(data)
			time = np.empty((data_set_size,))
			low = np.empty((data_set_size,))
			high = np.empty((data_set_size,))
			open_p = np.empty((data_set_size,))
			close_p = np.empty((data_set_size,))
			volume = np.empty((data_set_size,))

			for z in range(len(data)):
				
				time[z] = data[z][0][0]
				low[z] = data[z][0][1]
				high[z] = data[z][0][2]
				open_p[z] = data[z][0][3]
				close_p[z] = data[z][0][4]
				volume[z] = data[z][0][5]

"""
				time = (time - statistics.mean(time))/statistics.stdev(time)
				low = (low - statistics.mean(low))/statistics.stdev(low)
				high = (high - statistics.mean(high))/statistics.stdev(high)
				open_p = (open_p - statistics.mean(open_p))/statistics.stdev(open_p)
				close_p = (close_p - statistics.mean(close_p))/statistics.stdev(close_p)
				volume = (volume - statistics.mean(volume))/statistics.stdev(volume)
"""
				
				

			#time = np.append(time,["time"], axis = 0)
			#close_p = np.append(close_p, ["close_p"], axis = 0)
			#open_p = np.append(open_p, ["open_p"], axis = 0)

			combined_tuple = (time, low, high, open_p, close_p,volume)

			combined_data_set =  np.hstack(combined_tuple)
			#print("COMBINE DATA SET  :", combined_data_set)

			return combined_data_set