import pandas as pd 
import numpy as np
import torch

def dataloader_adult(train_path = "/Users/plumyu/Desktop/FZ_federated/FuzzyFL-main/data/adult.data", test_path = "/Users/plumyu/Desktop/FZ_federated/FuzzyFL-main/data/adult.test"):
	
	train_set = pd.read_csv(train_path)#, header = None)
	test_set = pd.read_csv(test_path)#, header = None)

	# assign columns' names
	col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'wage_class']
	train_set.columns = col_labels
	test_set.columns = col_labels

	# deal with missing values
	#将有？的行全部丢弃
	train_set = train_set.replace('?', np.nan).dropna()
	test_set = test_set.replace('?', np.nan).dropna()

	# replace the value of 'wage_class' in test_set with the identical ones in the train_set 
	#在test集中有句号
	test_set['wage_class'] = test_set.wage_class.replace({' <=50K.': ' <=50K', ' >50K.': ' >50K'})

	# Encode categorical features
	combined_set = pd.concat([train_set, test_set], axis = 0)
	for feature in combined_set.columns:
		if combined_set[feature].dtype == 'object':
			combined_set[feature] = pd.Categorical(combined_set[feature]).codes

	combined_set.rename(columns = {'wage_class':'target'}, inplace = True)

	train_set = combined_set[:train_set.shape[0]]
	test_set = combined_set[train_set.shape[0]:]

	return train_set, test_set

def clip_grad(grad, clip):
	"""
	Gradient clipping
	"""
	g_shape = grad.shape
	grad.flatten()
	grad = grad / np.max((1, float(torch.norm(grad, p=3)) / clip))
	grad.view(g_shape)
	return grad