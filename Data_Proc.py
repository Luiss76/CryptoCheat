import bitfinex
import time
from time import ctime
import numpy as np
import json
import matplotlib.pyplot as plt
import statistics
import sys
import json                        # parse json into a list
import pandas as pd                # working with data frames
import datetime as dt              # working with dates
from sklearn.preprocessing import MinMaxScaler
				
class Data_Set_Proc:

	def __init__(self):
		self.scaler_close = MinMaxScaler(feature_range = (0,1))
		pass

	def dissect_data(self,array_size,interval):
		self.dataset = []
		self.ans_data = []
		combined_data_set = (self.time[0:array_size], self.low[0:array_size],self.high[0:array_size],self.open_p[0:array_size],self.close_p[0:array_size],self.volume[0:array_size])
		combined_data_set =  np.hstack(combined_data_set)
		self.dataset.append(combined_data_set)
		no_range = int(len(self.low[array_size:])/interval)
		for i in range(no_range):

			combined_data_set =(self.time[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.low[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.high[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.open_p[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.close_p[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.volume[array_size+(i*interval):(array_size+(i*interval)+array_size)])
			combined_data_set = np.hstack(combined_data_set)
			self.dataset.append(combined_data_set)
			#print("i :",i)
			self.ans_data.append(self.dataset[i+1][(4*array_size)+(array_size-interval):(4*array_size)+(array_size-interval)+interval])



		self.dataset = self.dataset[1:-3]
		self.ans_data = self.ans_data[1:-2]
		print("\n")
		#print("data set :", 4473, " \n", self.dataset[4473][4*array_size:(4*array_size)+array_size])
		
		#print("actual results :", 4473, " \n", self.ans_data[4473])
		#print("first close for close :", self.close_p[array_size+(0*interval):(array_size+(0*interval)+array_size)],"second data : ",self.close_p[array_size+(1*interval):(array_size+(1*interval)+array_size)])
		#print("first data for close :", self.dataset[0][4*array_size:(4*array_size)+array_size],"second data : ",self.dataset[1][4*array_size:(4*array_size)+array_size])
		#print("first data next for close : ",self.dataset[0][(4*array_size)+(array_size-interval):(4*array_size)+(array_size-interval)+interval])
		#print("second data next for close : ",self.dataset[1][(4*array_size)+(array_size-interval):(4*array_size)+(array_size-interval)+interval])
		
		#print("dataset :",len(self.dataset[740]))

	def produce_trest(self,training_portion,validation_portion):
		training_size = int(len(self.dataset)*training_portion/100)
		ans_training_size = int(len(self.ans_data)*training_portion/100)
		self.data = self.dataset[0:training_size]
		self.ans_data_ori = self.ans_data[0:ans_training_size]
		self.testing_data = self.dataset[training_size:]
		self.ans_testing_data = self.dataset[ans_training_size:]

		validation_size = int(len(self.data)*validation_portion/100)
		ans_validation_size = int(len(self.ans_data_ori)*validation_portion/100)
		self.validation_data = self.data[0:validation_size]
		self.ans_validation_data = self.ans_data_ori[0:ans_validation_size]
		self.training_data = self.data[validation_size:]
		self.ans_training_data = self.ans_data_ori[ans_validation_size:]
		return self.training_data,self.validation_data,self.testing_data
	

	def get_import_data(self,address):
		print("inside")
		with open(address,) as f:
			self.data = json.load(f)
			print(self.data[0][0])
			self.data_set_size = len(self.data[0])
			self.time = np.empty((self.data_set_size,))
			self.low = np.empty((self.data_set_size,))
			self.high = np.empty((self.data_set_size,))
			self.open_p = np.empty((self.data_set_size,))
			self.close_p = np.empty((self.data_set_size,))
			self.volume = np.empty((self.data_set_size,))

		for z in range(len(self.data[0])):

				self.time[z] = self.data[0][z]/1000
				self.open_p[z] = self.data[1][z]
				self.close_p[z] = self.data[2][z]
				self.high[z] = self.data[3][z]
				self.low[z] = self.data[4][z]
				self.volume[z] = self.data[5][z]

		scaler = MinMaxScaler(feature_range = (0,1))
		self.scaled_open = scaler.fit_transform(self.open_p.reshape(-1,1))
		self.scaled_close = self.scaler_close.fit_transform(self.close_p.reshape(-1,1))
		self.scaled_high = scaler.fit_transform(self.high.reshape(-1,1))
		self.scaled_low = scaler.fit_transform(self.low.reshape(-1,1))
		self.scaled_volume = scaler.fit_transform(self.volume.reshape(-1,1))

		return self.time,self.scaled_open,self.scaled_close,self.scaled_high,self.scaled_low,self.scaled_volume

	def get_data_set_bitfinex(self,s_year,n_year,s_month,n_month):
		d0 = dt.date(s_year, s_month, 1)
		d1 = dt.date(n_year, n_month, 1)
		interval = d1 - d0
		print("how many days : ",d1 -d0)
		api_v2 = bitfinex.bitfinex_v2.api_v2()
		df = []
		empty_array = []
		time_array = []
		open_array = []
		close_array = []
		high_array = []
		low_array = []
		volume_array = []

		# Define query parameters
		pair = 'BTCUSD' # Currency pair of interest
		TIMEFRAME = '5m'#,'4h','1h','15m','1m'
		#interval =  ((n_month - s_month)* 10)
		# Convert list of data to pandas dataframe
		names = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
		for i in range(interval.days):
			print("i :", i)
			# Define the start date
			t_start = dt.datetime(s_year, s_month, 1, 0, 0) + dt.timedelta(days = i)
			t_start = time.mktime(t_start.timetuple()) * 1000

			# Define the end date
			t_stop = dt.datetime(s_year, s_month, 1, 0, 0) + dt.timedelta(days = i+1)
			t_stop = time.mktime(t_stop.timetuple()) * 1000

			# Download OHCL data from API
			result = api_v2.candles(symbol=pair, interval=TIMEFRAME, limit=1000, start=t_start, end=t_stop)
			a = pd.DataFrame(result, columns=names)
			df.append(a)

		#df['Date'] = pd.to_datetime(df['Date'], unit='ms')
		#print("HMM :", len(df[0]))
		for j in range(len(df)):
			for i in range(len(df[j])):
				time_array.append(df[j].Date[i])
				open_array.append(df[j].Open[i])
				close_array.append(df[j].Close[i])
				high_array.append(df[j].High[i])
				low_array.append(df[j].Low[i])
				volume_array.append(df[j].Volume[i])
		#print("close :", len(close_array))

		#print("time :",dt.datetime.fromtimestamp(1609603200000/1000))
		#for i in range(len(time_array)):
		#	time_array[i] = dt.datetime.fromtimestamp(time_array[i]/1000)
		#plt.plot(time_array,close_array)
		#plt.show()
		return time_array,open_array,close_array,high_array,low_array,volume_array
	"""
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
	def get_time(self):
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
		"""