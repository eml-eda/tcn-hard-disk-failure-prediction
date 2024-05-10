import os
import pandas as pd
import datetime
import numpy as np
from numpy import *
import math
import pickle
from scipy.stats.stats import pearsonr
import sys
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
from Networks_pytorch import *
from Dataset_manipulation import *
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



def classification(X_train, Y_train, X_test, Y_test, classifier, metric, **args):
	"""
	Perform classification using the specified classifier.

	Parameters:
	- X_train (array-like): Training data features.
	- Y_train (array-like): Training data labels.
	- X_test (array-like): Test data features.
	- Y_test (array-like): Test data labels.
	- classifier (str): The classifier to use. Options: 'RandomForest', 'TCN', 'LSTM'.
	- metric (str): The metric to evaluate the classification performance.
	- **args: Additional arguments specific to each classifier.

	Returns:
	- None
	"""
	from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, precision_score
	from sklearn.utils import shuffle
	print('Classification using {} is starting'.format(classifier))
	Y_test_real = []
	prediction = []
	if classifier == 'RandomForest':
		from sklearn.ensemble import RandomForestClassifier
		X_train, Y_train = shuffle(X_train, Y_train)
		model = RandomForestClassifier(n_estimators=30, min_samples_split=10, random_state=3)
		model.fit(X_train[:, :], Y_train)
		prediction = model.predict(X_test)
		Y_test_real = Y_test
		report_metrics(Y_test_real, prediction, metric)
	elif classifier == 'TCN':
		net_train_validate(args['net'], args['optimizer'], X_train, Y_train, X_test, Y_test, args['epochs'], args['batch_size'], args['lr'])
	elif classifier == 'LSTM':
		train_dataset = FPLSTMDataset(X_train, Y_train)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=FPLSTM_collate)
		test_dataset = FPLSTMDataset(X_test, Y_test.values)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=FPLSTM_collate)
		net_train_validate_LSTM(args['net'], args['optimizer'], train_loader, test_loader, args['epochs'], X_test.shape[0], Xtrain.shape[0], args['lr'])
		pass

if __name__ == '__main__':
	features = {'Xiao_et_al':['date','serial_number','model','failure','smart_1_normalized','smart_5_normalized','smart_5_raw','smart_7_normalized','smart_9_raw',\
					'smart_12_raw','smart_183_raw','smart_184_normalized','smart_184_raw','smart_187_normalized','smart_187_raw',\
					'smart_189_normalized','smart_193_normalized','smart_193_raw','smart_197_normalized','smart_197_raw','smart_198_normalized','smart_198_raw','smart_199_raw'],\
				'iSTEP':['date','serial_number','model','failure','smart_5_raw','smart_3_raw','smart_10_raw',\
					'smart_12_raw','smart_4_raw','smart_194_raw','smart_1_raw','smart_9_raw',\
					'smart_192_raw','smart_193_raw','smart_197_raw','smart_198_raw','smart_199_raw']}
	#model = 'ST4000DM000'
	# here you can select the model. This is the one tested.
	model = 'ST3000DM001'
	#years = ['2016','2017','2018']
	years = ['2014','2015','2016','2017','2018']
	# many parameters that could be changed, both for unbalancing, for networks and for features.
	windowing = 1
	min_days_HDD = 115
	days_considered_as_failure = 7
	test_train_perc = 0.3
	# type of oversampling
	oversample_undersample = 2
	# balancing factor (major/minor = balancing_normal_failed)
	balancing_normal_failed = 20
	history_signal = 32
	# type of classifier
	classifier = 'LSTM'
	# if you extract features for RF for example. Not tested
	perform_features_extraction = False
	CUDA_DEV = "0"
	# if automatically select best features
	ranking = 'Ok'
	num_features = 18
	overlap = 1
	try:
		df = pd.read_pickle('../temp/' + model +'_Dataset_windowed_' + str(history_signal) +'_rank_'+ranking + '_' +str(num_features)+ '_overlap_'+str(overlap)+'.pkl')
	except:
		if ranking == 'None':
			df = import_data(years= years, model = model, name = 'iSTEP', features=features)
		else:
			df = import_data(years = years, model = model, name = 'iSTEP')
		print(df.head())
		for column in list(df):
			missing = round(df[column].notna().sum() / df.shape[0] * 100, 2)
			print('{:.<27}{}%'.format(column, missing))
		# drop bad HDs
		bad_missing_hds, bad_power_hds, df = filter_HDs_out(df, min_days = min_days_HDD, time_window='30D', tolerance=30)
		df['y'], df['val'] = Y_target(df, days=days_considered_as_failure, window = history_signal) # define RUL piecewise
		if ranking is not 'None':
			df = feature_selection(df, num_features)
		print('Used features')
		for column in list(df):
			print('{:.<27}'.format(column,))	
		## -------- ##
		# random: stratified without keeping timw
		# hdd --> separate different hdd (need FIXes)
		# temporal --> separate by time (need FIXes)
	Xtrain, Xtest, ytrain, ytest = dataset_partitioning(df, model, overlap = overlap, rank = ranking,num_features= num_features, technique = 'random', test_train_perc = test_train_perc, windowing = windowing, window_dim =history_signal, resampler_balancing = balancing_normal_failed, oversample_undersample = oversample_undersample)
	####### CLASSIFIER PARAMETERS #######
	if classifier == 'RandomForest':
		pass
	elif classifier == 'TCN':
		os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV
		batch_size = 256
		lr = 0.001	
		num_inputs = Xtrain.shape[1]
		net, optimizer = init_net(lr, history_signal, num_inputs)
		epochs = 200
	elif classifier == 'LSTM':
		lr = 0.001
		batch_size = 256
		epochs = 300	
		dropout = 0.1
		#hidden state sizes (from [14])
		lstm_hidden_s = 64
		fc1_hidden_s = 16	
		num_inputs = Xtrain.shape[1]	
		net = FPLSTM(lstm_hidden_s,fc1_hidden_s,num_inputs,2,dropout)
		net.cuda()
		optimizer = optim.Adam(net.parameters(), lr=lr)
	## ---------------------------- ##
	if perform_features_extraction == True:
		Xtrain = feature_extraction(Xtrain)
		Xtest = feature_extraction(Xtest)
	if classifier == 'RandomForest' and windowing==1:
		Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1]*Xtrain.shape[2])
		Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1]*Xtest.shape[2])

	try:
		classification(X_train = Xtrain, Y_train = ytrain, X_test = Xtest, Y_test = ytest, classifier = classifier, metric = ['RMSE', 'MAE', 'FDR','FAR','F1','recall', 'precision'], net = net, optimizer = optimizer, epochs = epochs, batch_size = batch_size, lr= lr)
	except:
		classification(X_train = Xtrain, Y_train = ytrain, X_test = Xtest, Y_test = ytest, classifier = classifier, metric = ['RMSE', 'MAE', 'FDR','FAR','F1','recall', 'precision'])