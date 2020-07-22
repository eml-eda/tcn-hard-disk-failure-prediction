import pandas as pd
import datetime
import numpy as np
from numpy import *
import math
import pickle
from scipy.stats.stats import pearsonr
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
import glob
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

## here there are many functions used inside Classification.py


def plot_feature(dataset):
	X = dataset['X']
	Y = dataset['Y']
	feat = X[:,10]
	np.arange(len(feat))
	fig,ax = plt.subplots()
	ax.scatter(np.arange(len(feat[Y==0])),feat[Y==0], color ='C0', label='good HDD')
	ax.scatter(np.arange(len(feat[Y==0]),len(feat)),feat[Y==1],color ='C1',label='failed HDD')
	ax.set_ylabel('SMART feature [#]', fontsize=14, fontweight='bold', color = 'C0')
	ax.set_xlabel('time points / hdd', fontsize=14, fontweight='bold')
	legend_properties = {'weight':'bold'}
	plt.legend(fontsize=12, prop = legend_properties)
	plt.show()

def plot_hdd(X, fail,prediction):
	fig,ax = plt.subplots()	
	features = {'Xiao_et_al':['date','failure','smart_1_normalized','smart_5_normalized','smart_5_raw','smart_7_normalized','smart_9_raw',\
					'smart_12_raw','smart_183_raw','smart_184_normalized','smart_184_raw','smart_187_normalized','smart_187_raw',\
					'smart_189_normalized','smart_193_normalized','smart_193_raw','smart_197_normalized','smart_197_raw','smart_198_normalized','smart_198_raw','smart_199_raw']}
	k=0
	for i in [1,2,7,8,9,10]:
		ax.plot(np.arange(X.shape[0]),X[:,i]+0.01*k, label=features['Xiao_et_al'][i+2])
		k+=1
	ax.set_ylabel('SMART features value [#]', fontsize=14, fontweight='bold', color = 'C0')
	ax.set_xlabel('time points', fontsize=14, fontweight='bold')
	legend_properties = {'weight':'bold'}
	plt.legend(fontsize=12, prop = legend_properties)
	plt.title('The HDD is {} (failed=1/good=0) predicted as {}'.format(fail,prediction))
	plt.show()

def pandas_to_3dmatrix(read_dir, model, years, dataset_raw):
	join_years = '_'
	for year in years:
		join_years = join_years + year + '_'
	name_file = 'Matrix_Dataset' + join_years +'.pkl'
	try :
		with open(os.path.join(read_dir,name_file), 'rb') as handle:
			dataset = pickle.load(handle)
		print('Matrix 3d {} already present'.format(name_file))
	except:
		print('Creating matrix 3d {}'.format(name_file))
		row_valids = []
		rows_dim = len(dataset_raw['failure'])
		feat_dim=len(dataset_raw.keys()[2:])
		### removing HDD with some features always nan
		for k in np.arange(rows_dim):
			row = dataset_raw.iloc[k]
			invalid = False
			for i in np.arange(2,len(row.keys())):
				if len(row[i]) == sum(math.isnan(x) for x in row[i]):
					invalid=True
			if invalid == True:
				row_valids.append(False)
			else:
				row_valids.append(True)
		dataset_raw = dataset_raw[row_valids]
		rows_dim = len(dataset_raw['failure'])
		# computing Max time stamps in an hdd
		max_len = 0
		print('Computing maximum number of timestamps of one hdd')
		for k in np.arange(rows_dim):
			row = dataset_raw.iloc[k]
			if max_len < len(row['date']):
				max_len =len(row['date'])
		matrix_3d = []
		for k in np.arange(rows_dim):
			row = dataset_raw.iloc[k]
			print('Analizing HD number {} \r'.format(k), end="\r")
			row = dataset_raw.iloc[k]
			hd = []
			for j,features in enumerate(row.keys()[1:]):
				hd.append(row[features])
			hd = np.asarray(hd)
			hd_padded = np.concatenate((hd,np.ones((hd.shape[0],max_len-hd.shape[1]))*2),axis=1)
			matrix_3d.append(hd_padded.T)
		matrix_3d = np.asarray(matrix_3d)

		## for debugging
		good = 0
		failed = 0
		disk_type = []
		for k in np.arange(matrix_3d.shape[0]):
			row = matrix_3d[k,:,0]
			try: 
				np.where(row==1)[0][0]
				failed+=1
			except:
				good +=1
		print('There are {} disk goods and {} failed in the dataset'.format(good,failed))	
		dataset = {'matrix': matrix_3d}
		with open(os.path.join(read_dir,name_file), 'wb') as handle:
			pickle.dump(dataset,handle)
	return dataset

def matrix3d_to__datasets(matrix, window = 1, divide_hdd = 1, training_percentage = 0.7, lambda_unbalancing = 3):
	#np.random.shuffle(matrix)
	name_file = 'Final_Dataset.pkl'
	read_dir = os.path.dirname('../data_input/')
	try :
		with open(os.path.join(read_dir,name_file), 'rb') as handle:
			dataset = pickle.load(handle)
		print('Dataset {} already present'.format(name_file))
	except:
		X_matrix = matrix[:,:,:]
		Y_matrix = np.zeros(X_matrix.shape[0])
		for hdd in np.arange(X_matrix.shape[0]):
			try:
				np.where(X_matrix[hdd,:,0]==1)[0][0]
				Y_matrix[hdd] = 1
			except:
				pass
		print('Failed hard disk = {}'.format(sum(Y_matrix)))
		if divide_hdd == 1:
			from sklearn.model_selection import train_test_split
			X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(X_matrix, Y_matrix, stratify=Y_matrix, test_size=1-training_percentage)
			### creating dataset per window.
			### training set creation
			first = 1
			print('Processing training dataset')
			for hdd_number in np.arange(y_train_hd.shape[0]):
				print('Analizing HD number {} \r'.format(hdd_number), end="\r")
				if y_train_hd[hdd_number] == 1:
					temp = X_train_hd[hdd_number,:np.where(X_train_hd[hdd_number,:,0]==1)[0][0],1:]
					if first:	
						X_train = temp
						Y_train = np.concatenate((np.zeros(temp.shape[0]-7), np.ones(7)))
						first = 0
					else:
						X_train = np.concatenate((X_train,temp))
						try:
							Y_train = np.concatenate((Y_train,np.zeros(temp.shape[0]-7), np.ones(7)))
						except:
							Y_train = np.concatenate((Y_train,np.ones(temp.shape[0])))
				else:
					try:
						temp = X_train_hd[hdd_number,:(np.where(X_train_hd[hdd_number,:,0]==2)[0][0]-1-7),1:]
					except:
						temp = X_train_hd[hdd_number,:-7,1:]
					if first:	
						X_train = temp
						Y_train = (np.zeros(temp.shape[0]))
						first = 0
					else:
						X_train = np.concatenate((X_train,temp))
						try:
							Y_train = np.concatenate((Y_train,np.zeros(temp.shape[0])))
						except:
							import pdb;pdb.set_trace()
			first = 1
			print('Processing test dataset')
			for hdd_number in np.arange(y_test_hd.shape[0]):
				print('Analizing HD number {} \r'.format(hdd_number), end="\r")
				if y_test_hd[hdd_number] == 1:
					temp = X_test_hd[hdd_number,:np.where(X_test_hd[hdd_number,:,0]==1)[0][0],1:]
					if first:	
						X_test = temp
						Y_test = np.concatenate((np.zeros(temp.shape[0]-7), np.ones(7)))
						HD_number_test = np.zeros(temp.shape[0])
						first = 0
					else:
						X_test = np.concatenate((X_test,temp))
						HD_number_test = np.concatenate((HD_number_test,np.ones(temp.shape[0])*hdd_number))
						try:
							Y_test = np.concatenate((Y_test,np.zeros(temp.shape[0]-7), np.ones(7)))
						except:
							Y_test = np.concatenate((Y_test,np.ones(temp.shape[0])))
				else:
					try:
						temp = X_test_hd[hdd_number,:(np.where(X_test_hd[hdd_number,:,0]==2)[0][0]-1-7),1:]
					except:
						temp = X_test_hd[hdd_number,:-7,1:]
					if first:	
						X_test = temp
						Y_test = (np.zeros(temp.shape[0]))
						HD_number_test = np.zeros(temp.shape[0])
						first = 0
					else:
						X_test = np.concatenate((X_test,temp))
						HD_number_test = np.concatenate((HD_number_test,np.ones(temp.shape[0])*hdd_number))
						try:
							Y_test = np.concatenate((Y_test,np.zeros(temp.shape[0])))	
						except:
							import pdb;pdb.set_trace()
			Y_train = Y_train[~np.isnan(X_train).any(axis=1)]
			X_train = X_train[~np.isnan(X_train).any(axis=1)]
			Y_test = Y_test[~np.isnan(X_test).any(axis=1)]
			HD_number_test = HD_number_test[~np.isnan(X_test).any(axis=1)]
			X_test = X_test[~np.isnan(X_test).any(axis=1)]
			if oversample_undersample == 0:
				from imblearn.under_sampling import RandomUnderSampler
				rus = RandomUnderSampler(1/resampler_balancing,random_state=42)
			else:
				from imblearn.over_sampling import SMOTE
				rus = SMOTE(1/resampler_balancing,random_state=42)
			X_train, Y_train = rus.fit_resample(X_train, Y_train)
			dataset = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test, 'HDn_test': HD_number_test }
			with open(os.path.join(read_dir,name_file), 'wb') as handle:
				pickle.dump(dataset,handle)
	return dataset

def import_data(years, model,name, **args):
	""" Import hard drive data from csvs on disk.
	
	:param quarters: List of quarters to import (e.g., 1Q19, 4Q18, 3Q18, etc.)
	:param model: String of the hard drive model number to import.
	:param columns: List of the columns to import.
	:return: Dataframe with hard drive data.
	
	"""
	years_list = ''
	for y in years:
		years_list = years_list+ '_' + y
	try:
		feat = args['features']
		file = '../temp/' + model + years_list+'.pkl'	
	except:
		file = '../temp/' + model + years_list+'_all.pkl'	
	try:
		df = pd.read_pickle(file)
	except:
		cwd = os.getcwd()
		df = pd.DataFrame()
		for y in years:
			print('Analizing year {} \r'.format(y), end="\r")
			try:
				data = pd.concat([pd.read_csv(f, header=0, usecols=args['features'][name], parse_dates=['date'])[pd.read_csv(f, header=0, usecols=args['features'][name], parse_dates=['date']).model == model] for f in glob.glob(cwd + '/../../HDD_dataset/' + y + '/*.csv')], ignore_index=True)
			except:
				data = pd.concat([pd.read_csv(f, header=0, parse_dates=['date'])[pd.read_csv(f, header=0, parse_dates=['date']).model == model] for f in glob.glob(cwd + '/../../HDD_dataset/' + y + '/*.csv')], ignore_index=True)
						#data = data[data.model == model]
			data.drop(columns=['model'], inplace=True)
			data.failure = data.failure.astype('int')
			#data.smart_9_raw = data.smart_9_raw / 24.0 # convert power-on hours to days
			df = pd.concat([df, data])
	        
		df.reset_index(inplace=True, drop=True)
		df = df.set_index(['serial_number', 'date']).sort_index()
		df.to_pickle(file)
	return df


def filter_HDs_out(df, min_days, time_window, tolerance):
	""" Find HDs with an excessive amount of missing values.
	
	:param df: Input dataframe.
	:param min_days: Minimum number of days HD needs to have been powered on.
	:param time_window: Size of window to count missing values (e.g., 7 days as '7D', or five days as '5D')
	:param tolerance: Maximum number of days allowed to be missing within time window.
	:return: List of HDs with excessive amount of missing values
	
	"""
	pcts = df.notna().sum() / df.shape[0] * 100
	cols = pcts < 45.0 # identify columns to remove True / False
	cols = cols[list(cols)] # select column names with 'True' value
	cols = cols[list(cols)].reset_index()
	cols = list(cols['index']) # generate a list of column names to remove
	df.drop(cols, axis=1, inplace=True) # drop columns
	df = df.dropna()
	bad_power_hds = []
	for serial_num, inner_df in df.groupby(level=0): # identify HDs with too few power-on days.
		if len(inner_df) < min_days:
			bad_power_hds.append(serial_num)
		else:
			pass
        
	bad_missing_hds = []
	for serial_num, inner_df in df.groupby(level=0): # indentify HDs with too many missing values.
		inner_df = inner_df.droplevel(level=0)
		inner_df = inner_df.asfreq('D')
		n_missing = max(inner_df.isna().rolling(time_window).sum().max())

		if n_missing >= tolerance:
			bad_missing_hds.append(serial_num)
		else:
			pass
	bad_hds = set(bad_missing_hds + bad_power_hds)
	hds_remove = len(bad_hds)
	hds_total = len(df.reset_index().serial_number.unique())
	print('Total HDs: {}    HDs removed: {} ({}%)'.format(hds_total, hds_remove, round(hds_remove/hds_total * 100, 2)))

	df = df.drop(bad_hds, axis=0)

	num_fail = df.failure.sum()
	num_not_fail = len(df.reset_index().serial_number.unique()) - num_fail
	pct_fail = num_fail / (num_not_fail + num_fail) * 100 

	print('{} failed'.format(num_fail))
	print('{} did not fail'.format(num_not_fail))
	print('{:5f}% failed'.format(pct_fail))
	return bad_missing_hds, bad_power_hds,df

def interpolate_ts(df, method='linear'):
    
    """ Interpolate hard drive Smart attribute time series.
    
    :param df: Input dataframe.
    :param method: String, interpolation method.
    :return: Dataframe with interpolated values.
    """
    
    interp_df = pd.DataFrame()

    for serial_num, inner_df in df.groupby(level=0):
        inner_df = inner_df.droplevel(level=0)
        inner_df = inner_df.asfreq('D') 
        inner_df.interpolate(method=method, axis=0, inplace=True)
        inner_df['serial_number'] = serial_num
        inner_df = inner_df.reset_index()

        interp_df = pd.concat([interp_df, inner_df], axis=0)

    df = interp_df.set_index(['serial_number', 'date']).sort_index()
    
    return df

def Y_target(df, days, window):
    pred_list = np.asarray([])
    valid_list = np.asarray([])
    i=0
    for serial_num, inner_df in df.groupby(level=0):
        print('Analizing HD {} number {} \r'.format(serial_num,i), end="\r")
        slicer_val = len(inner_df)  # save len(df) to use as slicer value on smooth_smart_9 
        i+=1
        if inner_df.failure.max() == 1:
            predictions = np.concatenate((np.zeros(slicer_val-days), np.ones(days)))
            valid = np.concatenate((np.zeros(slicer_val-days-window), np.ones(days+window)))
        else:
            predictions = np.zeros(slicer_val)
            valid = np.zeros(slicer_val)
        pred_list = np.concatenate((pred_list,predictions))
        valid_list = np.concatenate((valid_list,valid))
    print('A')
    pred_list = np.asarray(pred_list)
    valid_list = np.asarray(valid_list)
    return pred_list, valid_list

def arrays_to_matrix(X, wind_dim):
	X_new = X.reshape(X.shape[0],int(X.shape[1]/wind_dim),wind_dim)
	return X_new

def feature_extraction(X):
	print('Extracting Features')
	samples, features, dim_window = X.shape
	X_feature = np.ndarray((X.shape[0],X.shape[1], 4))
	print('Sum')
	X_feature[:,:,0] = np.sum((X), axis = 2)
	print('Min')
	X_feature[:,:,1] = np.min((X), axis = 2)
	print('Max')
	X_feature[:,:,2] = np.max((X), axis = 2)
	print('Similar slope')
	X_feature[:,:,3] = (np.max((X), axis = 2) - np.min((X), axis = 2))/dim_window
	from sklearn.linear_model import LinearRegression
	'''
	print('Slope')
	for s in np.arange(samples):
		for f in np.arange(features):
			model = LinearRegression()
			model.fit(np.arange(X.shape[2]).reshape(-1,1),X[s,f,:])
			X_feature[s,f,3] = model.coef_
			X_feature[s,f,4] = model.intercept_
	'''
	return X_feature

def factors(n):
	factors = []
	while n > 1:
		for i in range(2, n + 1):
			if n % i == 0:
				n /= i
				n = int(n)
				factors.append(i)
				break
	return factors

def under_sample(df, down_factor):
	indexes = df.y.rolling(down_factor).max()[((len(df)-1)%down_factor):-7:down_factor].index.tolist()
	return indexes

def dataset_partitioning(df, model, overlap = 0, rank = 'None', num_features = 10, technique = 'random', test_train_perc = 0.2, windowing = 1, window_dim = 5, resampler_balancing = 5,oversample_undersample = 0):
	df.reset_index(inplace=True)
	mms = MinMaxScaler(feature_range=(0, 1))
	temporal = df.loc[:, ['serial_number', 'date','failure', 'y','val']] 
	df = df.drop(columns=['serial_number', 'date','failure', 'y','val'], axis=1)
	df = pd.DataFrame(mms.fit_transform(df), columns= df.columns, index=df.index)
	df[['serial_number', 'date','failure', 'y','val']] = temporal
	loaded = 0
	if windowing == 1:
		try:
			windowed_df = pd.read_pickle('../temp/' + model +'_Dataset_windowed_' + str(window_dim) +'_rank_'+rank + '_' +str(num_features)+ '_overlap_'+str(overlap)+'.pkl')
			print('Loading the windowed dataset')
			loaded = 1
			cols = []
			count = {}
			for column in df.columns:
				count[column]=1
			for column in windowed_df.columns:
				if column in cols:
					cols.append(f'{column}_{count[column]}')
					count[column]+=1
					continue
				cols.append(column)
			windowed_df.columns = cols
			windowed_df.sort_index(axis=1,inplace=True)	
		except:
			print('Windowing the df')
			#df = df.set_index(['serial_number']).sort_index()
			windowed_df = df
			df1 = df
			cols = []
			count = {}
			for column in df.columns:
				count[column]=1
			if overlap == 1:
				for i in np.arange(window_dim-1):
					print('Concatenating time - {} \r'.format(i), end="\r")
					windowed_df = pd.concat([df.shift(i+1),windowed_df],axis=1)
			elif overlap == 0:
				window_dim_divisors = factors(window_dim)
				k = 0
				down_factor_old = 1
				serials = df.serial_number
				for down_factor in window_dim_divisors:					
					for i in np.arange(down_factor-1):
						k+=down_factor_old
						print('Concatenating time - {} \r'.format(k), end="\r")
						windowed_df = pd.concat([df.shift(i+1),windowed_df],axis=1)
					down_factor_old = down_factor_old * down_factor
					indexes = windowed_df.groupby(serials).apply(under_sample, down_factor)
					windowed_df = windowed_df.loc[np.concatenate(indexes.values.tolist(),axis=0), :]
					df = windowed_df
			else:
				window_dim_divisors = factors(window_dim)
				k = 0
				down_factor_old = 1
				serials = df.serial_number
				df_failed = df[df.val==1]
				windowed_df_failed = df_failed
				for i in np.arange(window_dim-1):
					print('Concatenating time failed - {} \r'.format(i), end="\r")
					windowed_df_failed = pd.concat([df_failed.shift(i+1),windowed_df_failed],axis=1)
				for down_factor in window_dim_divisors:					
					for i in np.arange(down_factor-1):
						k+=down_factor_old
						print('Concatenating time - {} \r'.format(k), end="\r")
						windowed_df = pd.concat([df.shift(i+1),windowed_df],axis=1)
					down_factor_old = down_factor_old * down_factor
					indexes = windowed_df.groupby(serials).apply(under_sample, down_factor)
					windowed_df = windowed_df.loc[np.concatenate(indexes.values.tolist(),axis=0), :]
					df = windowed_df
				windowed_df = windowed_df.append(windowed_df_failed)
				windowed_df.reset_index(inplace=True,drop=True)

			for column in windowed_df.columns:
				if column in cols:
					cols.append(f'{column}_{count[column]}')
					count[column]+=1
					continue
				cols.append(column)
			windowed_df.columns = cols
			windowed_df.sort_index(axis=1,inplace=True)	
			df = windowed_df
			print('C')
	print('Creating training and test dataset')
	if technique == 'random':
		if loaded == 1:
			df = windowed_df
		else:
			if windowing == 1:
				df['y'] = df['y_{}'.format(count['y']-1)]
				for i in np.arange(window_dim-1): 
					print('Dropping useless features of time - {} \r'.format(i), end="\r")
					df = df.drop(columns = ['serial_number_{}'.format(i+1), 'date_{}'.format(i+1),'failure_{}'.format(i+1), 'y_{}'.format(i+1), 'val_{}'.format(i+1)], axis=1)
				df = df.dropna()
				df = df.set_index(['serial_number', 'date']).sort_index()
				k=0
				for serial_num, inner_df in df.groupby(level=0):
					print('Computing index of invalid windows of HD number {} \r'.format(k), end="\r")
					k+=1
					#inner_df = inner_df.reset_index()
					if overlap==1:
						if k ==1:
							indexes = inner_df[:window_dim].index
						else:
							indexes = indexes.append(inner_df[:window_dim].index)
					else:
						if k ==1:
							indexes = inner_df[:1].index
						else:
							indexes = indexes.append(inner_df[:1].index)													
					#temp_df = pd.concat([temp_df, inner_df[window_dim:]], axis=0)
				print('C')
				print('Dropping invalid windows ')
				df = df.drop(indexes)
				df = df.reset_index()
				df.to_pickle('../temp/' + model +'_Dataset_windowed_'+ str(window_dim) +'_rank_'+rank + '_' + str(num_features)+ '_overlap_'+str(overlap)+'.pkl')
		y = df['y']
		df = df.drop(columns = ['serial_number', 'date','failure', 'y','val'], axis=1)
		X = df.values
		if windowing == 1:
			X = arrays_to_matrix(X, window_dim)
		Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,stratify=y,test_size=test_train_perc, random_state=42)
		if oversample_undersample == 0:
			from imblearn.under_sampling import RandomUnderSampler
			rus = RandomUnderSampler(1/resampler_balancing,random_state=42)
		else:
			from imblearn.over_sampling import SMOTE
			rus = SMOTE(1/resampler_balancing,random_state=42)
		if oversample_undersample != 2:			
			if windowing == 1:		
				dim1 = Xtrain.shape[1]
				dim2 = Xtrain.shape[2]
				Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2])
				ytrain = ytrain.astype(int)
				Xtrain, ytrain = rus.fit_resample(Xtrain, ytrain)
				Xtrain = Xtrain.reshape(Xtrain.shape[0],dim1,dim2)
				ytest = ytest.astype(int)
			else:
				Xtrain, ytrain = rus.fit_resample(Xtrain, ytrain)	
		else:
				ytrain = ytrain.astype(int)		
				ytest = ytest.astype(int)	
				ytrain=ytrain.values	
	elif technique == 'hdd':
		# define train and test sets while taking into account whether the drive failed or not
		np.random.seed(0)
		df = df.set_index(['serial_number', 'date']).sort_index()
		if windowing == 1:
			failed = [h[0] for h in list(df[df.y == 1].index)]
		else:
			failed = [h[0] for h in list(df[df.failure == 1].index)]
		failed = list(set(failed)) # unique list of all drives that failed

		not_failed = [h[0] for h in list(df.index) if h[0] not in failed]
		not_failed = list(set(not_failed)) # unique list of all drives that did not fail
		# generate list of failed HDs for test
		test_failed = list(np.random.choice(failed, size=int(len(failed) * test_train_perc)))
		test_not_failed = list(np.random.choice(not_failed, size=int(len(not_failed) * test_train_perc)))
		test = test_failed + test_not_failed 
		# make sure there is a variety of ages for testing and plotting
		# generate list of not failed HDs for test
		train = failed + not_failed # set train as full list of available HDs
		train = list(filter(lambda x: x not in test, train)) # filter HDs that will be used for testing
		# create train dataframe and ytrain
		df_train = df.loc[train, :].sort_index()
		df_test = df.loc[test, :].sort_index()
		df_train.reset_index(inplace=True)
		df_test.reset_index(inplace=True)
		ytrain = df_train.y
		ytest = df_test.y
		df_train = df_train.drop(columns = ['serial_number', 'date','failure', 'y'], axis=1)
		df_test = df_test.drop(columns = ['serial_number', 'date','failure', 'y'], axis=1)
		Xtrain = df_train.values
		Xtest = df_test.values
		if windowing == 1:
			Xtrain = arrays_to_matrix(Xtrain)
			Xtest = arrays_to_matrix(Xtest)
		if oversample_undersample == 0:
			from imblearn.under_sampling import RandomUnderSampler
			rus = RandomUnderSampler(1/resampler_balancing,random_state=42)
		else:
			from imblearn.over_sampling import SMOTE
			rus = SMOTE(1/resampler_balancing,random_state=42)
		dim1 = Xtrain.shape[1]
		dim2 = Xtrain.shape[2]
		Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2])
		if windowing == 1:		
			Xtrain, ytrain = rus.fit_resample(Xtrain, ytrain.astype(int))
			Xtrain = Xtrain.reshape(Xtrain.shape[0],dim1,dim2)
		else:
			Xtrain, ytrain = rus.fit_resample(Xtrain, ytrain)
	else:
		df = df.set_index(['date']).sort_index()
		y = df.y
		df = df.drop(columns = ['serial_number','failure', 'y'], axis=1)
		X = df.values
		if windowing == 1:
			Xtrain = X[:int(X.shape[0]*(1-test_train_perc)),:,:]
			Xtest = X[int(X.shape[0]*(1-test_train_perc)):,:,:]
		else:
			Xtrain = X[:int(X.shape[0]*(1-test_train_perc)),:]
			Xtest = X[int(X.shape[0]*(1-test_train_perc)):,:]
		ytrain = y[:int(X.shape[0]*(1-test_train_perc))]
		ytest = y[int(X.shape[0]*(1-test_train_perc)):]
		if oversample_undersample == 0:
			from imblearn.under_sampling import RandomUnderSampler
			rus = RandomUnderSampler(1/resampler_balancing,random_state=42)
		else:
			from imblearn.over_sampling import SMOTE
			rus = SMOTE(1/resampler_balancing,random_state=42)
		dim1 = Xtrain.shape[1]
		dim2 = Xtrain.shape[2]
		Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2])
		if windowing == 1:		
			Xtrain, ytrain = rus.fit_resample(Xtrain, ytrain.astype(int))
			Xtrain = Xtrain.reshape(Xtrain.shape[0],dim1,dim2)
		else:
			Xtrain, ytrain = rus.fit_resample(Xtrain, ytrain)	
	return Xtrain, Xtest, ytrain, ytest

def feature_selection(df, num_features):
	import scipy
	features = []
	p = []
	dict1 = {}
	print('Feature selection')
	import pdb;pdb.set_trace()
	import scipy
	scipy.stats.pearsonr(df['smart_4_raw'],df['smart_4_normalized'])
	for feature in df.columns:
		if 'raw' in feature:
			print('T-test for fature {} \r'.format(feature), end="\r")
			_ ,p_val = scipy.stats.ttest_ind(df[df['y']==0][feature], df[df['y']==1][feature], axis=0,  nan_policy='omit')
			dict1[feature] = p_val
	print('T')
	features = {k: v for k, v in sorted(dict1.items(), key=lambda item: item[1])}
	features = pd.DataFrame(features.items(),index=features.keys()).dropna()
	features = features[:num_features][0].values
	for feature in df.columns:
		if 'smart' not in feature:
			features = np.concatenate((features,np.asarray(feature).reshape(1,)))
	df= df[features]
	return df

if __name__ == '__main__':
	features = {'Xiao_et_al':['date','serial_number','model','failure','smart_1_normalized','smart_5_normalized','smart_5_raw','smart_7_normalized','smart_9_raw',\
					'smart_12_raw','smart_183_raw','smart_184_normalized','smart_184_raw','smart_187_normalized','smart_187_raw',\
					'smart_189_normalized','smart_193_normalized','smart_193_raw','smart_197_normalized','smart_197_raw','smart_198_normalized','smart_198_raw','smart_199_raw']}
	#dataset = dataset[features['Xiao_et_al']]
	model = 'ST3000DM001'
	years = ['2013','2014','2015','2016','2017']
	df = import_data(years, model, features)
	print(df.head())
	for column in list(df):
		missing = round(df[column].notna().sum() / df.shape[0] * 100, 2)
		print('{:.<27}{}%'.format(column, missing))
	# drop bad HDs
	
	bad_missing_hds, bad_power_hds, df = filter_HDs_out(df, min_days = 30, time_window='30D', tolerance=30)
	df['y'] = Y_target(df, days=7) # define RUL piecewise
	## -------- ##
	# random: stratified without keeping timw
	# hdd --> separate different hdd
	# temporal --> separate by time
	Xtrain, Xtest, ytrain, ytest = dataset_partitioning(df, technique = 'random')
	#method = 'linear'
	#df = interpolate_ts(df, method=method)
