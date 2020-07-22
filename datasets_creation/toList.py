import os
import pandas as pd
import datetime
import numpy as np
model = 'ST3000DM001'
database = pd.read_pickle('../temp/All_failed_appended_' + model +'.pkl')
base = pd.DataFrame(database.groupby('serial_number')['date'].apply(list))
failed = False
os.makedirs('../data_input', exist_ok=True)
for i,smart in enumerate(database.keys()[4:]):
	print('concatenating feature ' + str(i) + '/'+ str(database.keys()[2:].shape[0]))
	base = pd.concat([base, database.groupby('serial_number')[smart].apply(list)], axis=1)
if failed == True:
	base.to_pickle('../data_input/2013_2014_2015_2016_2017_failed_' + model + '.pkl')
else:
	base.to_pickle('../data_input/2013_2014_2015_2016_2017_all_' + model + '.pkl')
