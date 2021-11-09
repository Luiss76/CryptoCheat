import bitfinex
import time
from time import ctime
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics
import sys
import json                        # parse json into a list
import pandas as pd                # working with data frames
import datetime as dt              # working with dates
import calendar
from sklearn.preprocessing import MinMaxScaler
			
class Data_Set_Proc:

	def __init__(self):
		self.scaler_close = MinMaxScaler(feature_range = (0,1))
		self.scaler_open = MinMaxScaler(feature_range = (0,1))
		self.scaler_low = MinMaxScaler(feature_range = (0,1))
		self.scaler_high = MinMaxScaler(feature_range = (0,1))
		self.scaler_volume = MinMaxScaler(feature_range = (0,1))
		pass



	def normalise(self,ds,array_size):
		self.dataset = ds[0:len(ds)-array_size]
		self.vol_dataset = ds[len(ds)-array_size:]
		#print("shape of vol dataset:",np.shape(self.vol_dataset))
		#print("max val :",max(self.dataset))
		#print("min val :",min(self.dataset))

		#self.sdev = statistics.stdev(self.dataset)
		#self.mean = statistics.mean(self.dataset)
		#normalised_dataset = (self.dataset - self.mean)/self.sdev
		self.min_ds = min(self.dataset)
		self.max_ds = max(self.dataset)

		self.min_vol = min(self.vol_dataset)
		self.max_vol = max(self.vol_dataset)

		normalised_dataset = 2*((self.dataset - self.min_ds)/(self.max_ds - self.min_ds))-1
		normalised_dataset = np.append(normalised_dataset,2*((self.vol_dataset - self.min_vol)/(self.max_vol - self.min_vol))-1)

		#print("max value :", max(normalised_dataset))
		#print("min valie :", min(normalised_dataset))
		#print("shape of normalised dataset:",np.shape(normalised_dataset))

		return normalised_dataset

	def unormalise(self,ds):
		unormalised_dataset = (((ds + 1)*(self.max_ds - self.min_ds))/2)+self.min_ds
		#print("unormalised_dataset :",unormalised_dataset)
		return unormalised_dataset


	def dissect_data(self,array_size,interval,_open,_close,_low,_high,_volume):
		self.unscaled_ds = []
		self.ans_data = []

		self.open_p = _open
		self.close_p = _close
		self.low = _low
		self.high = _high
		self.volume = _volume

		#print("closing :", np.shape(self.close_p))
		

		unscaled_combined_ds = (self.close_p[0:array_size],self.low[0:array_size],self.high[0:array_size],self.open_p[0:array_size],self.volume[0:array_size])
		unscaled_combined_ds = np.hstack(unscaled_combined_ds)
		#self.unscaled_ds.append(unscaled_combined_ds)

		
		#self.ans_data.append(self.close_p[array_size:(array_size+interval)])
		
		
		

		#combined_data_set = (self.scaled_low[0:array_size],self.scaled_high[0:array_size],self.scaled_open[0:array_size],self.scaled_close[0:array_size],self.scaled_volume[0:array_size])
		#combined_data_set =  np.hstack(combined_data_set)
		#self.dataset.append(combined_data_set)

		no_range = int(len(self.low[array_size:])/interval)
		#print("no range :",no_range)

		for i in range(no_range):


			"""
			i: 0
					0:120
			i:1
					12:132
			"""
			unscaled_combined_ds = (self.close_p[interval*i:((interval*i)+array_size)],self.low[interval*i:((interval*i)+array_size)],self.high[interval*i:((interval*i)+array_size)],self.open_p[interval*i:((interval*i)+array_size)],self.volume[interval*i:((interval*i)+array_size)])
			#print("shape of unscaled ds :", np.shape(unscaled_combined_ds), "index :", i)
			#print("closing array :",np.shape((self.close_p[array_size*i:((array_size*i)+array_size)])))
			unscaled_combined_ds = np.hstack(unscaled_combined_ds)
			unscaled_combined_ds = unscaled_combined_ds.flatten()
			
			
			self.unscaled_ds.append(unscaled_combined_ds)
			self.ans_data.append(self.close_p[array_size+(i*interval):(array_size+(i*interval)+interval)])
			#self.ans_data.append(self.ds[i+1][(3*array_size)+(array_size-interval):(3*array_size)+(array_size-interval)+interval])

			#combined_data_set =(self.scaled_low[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.scaled_high[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.scaled_open[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.scaled_close[array_size+(i*interval):(array_size+(i*interval)+array_size)],self.scaled_volume[array_size+(i*interval):(array_size+(i*interval)+array_size)])
			#combined_data_set = np.hstack(combined_data_set) 
			#combined_data_set = combined_data_set.flatten()
			#self.dataset.append(combined_data_set)
			#print("i :",i)
			#self.ans_data.append(self.unscaled_ds[i+1][(3*array_size)+(array_size-interval):(3*array_size)+(array_size-interval)+interval])
				#print("shape of first answer data :", np.shape(self.ans_data))

		#print("closing array :", self.close_p[1080:1200+interval])
		#print("answer data:", self.ans_data[9])
		#print("usncaled ds :",self.unscaled_ds[9][:120])
		#print("shape of closing array :",np.shape(self.unscaled_ds))


		#self.unscaled_ds = self.unscaled_ds[1:-3]
		#self.ans_data = self.ans_data[1:-2]

		#print("data set :", 4473, " \n", self.dataset[4473][4*array_size:(4*array_size)+array_size])
		
		#print("actual results :", 4473, " \n", self.ans_data[4473])
		#print("first close for close :", self.close_p[array_size+(0*interval):(array_size+(0*interval)+array_size)],"second data : ",self.close_p[array_size+(1*interval):(array_size+(1*interval)+array_size)])
		#print("first data for close :", self.dataset[0][4*array_size:(4*array_size)+array_size],"second data : ",self.dataset[1][4*array_size:(4*array_size)+array_size])
		#print("first data next for close : ",self.dataset[0][(4*array_size)+(array_size-interval):(4*array_size)+(array_size-interval)+interval])
		#print("second data next for close : ",self.dataset[1][(4*array_size)+(array_size-interval):(4*array_size)+(array_size-interval)+interval])
		
		#print("dataset :",len(self.dataset[740]))

	def produce_trest(self,training_portion,validation_portion):
		training_size = int(len(self.unscaled_ds)*training_portion/100)
		ans_training_size = int(len(self.ans_data)*training_portion/100)

		self.data = self.unscaled_ds[0:training_size]
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
		with open(address,) as f:
			self.data = json.load(f)
			#print(self.data[0][0])
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

		#scaler = MinMaxScaler(feature_range = (0,1))
		#self.scaled_open = self.scaler_open.fit_transform(self.open_p.reshape(-1,1))
		#self.scaled_close = self.scaler_close.fit_transform(self.close_p.reshape(-1,1))
		#self.scaled_high = self.scaler_high.fit_transform(self.high.reshape(-1,1))
		#self.scaled_low = self.scaler_low.fit_transform(self.low.reshape(-1,1))
		#self.scaled_volume = self.scaler_volume.fit_transform(self.volume.reshape(-1,1))

		return self.time,self.open_p,self.close_p,self.high,self.low,self.volume
	

	def get_data(self,s_year,n_year,s_month,n_month,_timeframe):
		names = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']

		df = []
		empty_array = []
		time_array = []
		open_array = []
		close_array = []
		high_array = []
		low_array = []
		volume_array = []



		t_start = dt.datetime(s_year,s_month,1,0,0)
		end_date = calendar.monthrange(n_year,n_month)[1]
		t_end = dt.datetime(n_year,n_month,end_date,23,59)
		diff = t_end - t_start
		t_start = time.mktime(t_start.timetuple()) * 1000
		t_end = time.mktime(t_end.timetuple()) * 1000

		print("diff :",diff)

		api_v2 = bitfinex.bitfinex_v2.api_v2()
		result = api_v2.candles(symbol='BTCUSD', interval=_timeframe, limit=10000, start=t_start, end=t_end)
		result.reverse()
		print("result :",np.shape(result))

		df.append(pd.DataFrame(result,columns=names))
		for j in range(len(df)):
			for i in range(len(df[j])):
				#print("time :",df[j].Date[i])
				time_array.append(df[j].Date[i])
				open_array.append(df[j].Open[i])
				close_array.append(df[j].Close[i])
				high_array.append(df[j].High[i])
				low_array.append(df[j].Low[i])
				volume_array.append(df[j].Volume[i])

		for i in range(len(time_array)):
			print("i :", i,"time_array:",dt.datetime.fromtimestamp(time_array[i]/1000))

		return time_array,open_array,close_array,high_array,low_array,volume_array