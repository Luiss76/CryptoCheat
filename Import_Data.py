from Data_Proc import Data_Set_Proc
import json
from numpyencoder import NumpyEncoder


s_year = 2017
s_month = 1
n_year = 2017
n_month = 12
timeframe = '1h'
data_set_proc = Data_Set_Proc()
numpy_data = data_set_proc.get_data(s_year,n_year,s_month,n_month,timeframe)  #start year, end year, start month, end month
with open('historical_data/'+ timeframe+'_'+str(s_year) + '_' +str(s_month) +'_to_'+ str(n_year) + '_' +str(n_month) +'_data.json', 'w') as file:
    json.dump(numpy_data, file, indent=4, sort_keys=True,
              separators=(', ', ': '), ensure_ascii=False,
              cls=NumpyEncoder)