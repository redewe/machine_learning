#!/usr/bin/python
import warnings
warnings.filterwarnings('ignore')

#Import all libraries
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

#Import all sklearn libraries for machine learning functions
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, scale 
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

#Import functions for pretty tables and graphing
from tabulate import tabulate
import matplotlib.pyplot as plt



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 
				'total_payments', 'total_stock_value', 'salary', 'deferral_payments', 
				'exercised_stock_options', 'bonus', 'restricted_stock', 
				'restricted_stock_deferred', 'expenses', 'loan_advances', 'other',  
				'director_fees', 'deferred_income','long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

	
#DATA EXPLORATION	
#Explore the details around the data by using dataframes
df = pd.DataFrame.from_dict(data_dict, orient='index') 
df.index.name = "Names"
df.sort_index(axis=1, inplace=True)

print("No. of Data Points: ", len(df))	
print("No. of Features: ",len(df.columns))
print("No. of POI: ", df["poi"].sum())

print("##################")

#Clean up NaN values in data
df2 = df.replace("NaN", 0)
df_NaN = df.groupby("Names").apply(lambda column: (column == "NaN").sum())

#Print data in table format
def tabulate_data(df_obj):	
	data = []
	for feature in df_obj:
		try:		
			data.append([df_obj[feature].name, df_NaN[feature].sum(), "{:,}".format(round(df_obj[feature].mean(), 2)), "{:,}".format(df_obj[feature].max()), "{:,}".format(df_obj[feature].min())])
		except:
			data.append([df_obj[feature].name, df_NaN[feature].sum(), "N/A", "N/A", "N/A"])
	print(tabulate(data, ["Feature", "No. of NaNs", "Mean", "Max.", "Min."], tablefmt="rst"))
	print("##################")

tabulate_data(df2)	
	
#Filter DF for POI only
df_poi = df2[df2['poi'] == 1]
df_poi = df_poi[features_list]
df_NaN = df[df['poi'] == 1].groupby("Names").apply(lambda column: (column == "NaN").sum())
tabulate_data(df_poi)	


### Task 2: Remove outliers

#Find the email address of the max value
df_new = df2[features_list]
email = df_new[df_new['total_payments'] == df_new['total_payments'].max()].index[0]
print("Email Address with Max Value :", email)

#Drop Outlier 1 - Total value
df_new = df_new[df_new.index != email]
data_dict.pop( "TOTAL", None ) 

#CREATE SCATTER PLOT
email = df_new[df_new['total_payments'] == df_new['total_payments'].max()].index[0]
print("Email Address with Max Value #2 :", email)

def scatter_plot(imgname):
	fig, ax = plt.subplots()
	ax.margins(0.05)
	groups = df_new.groupby('poi') 
	for name, group in groups:
		ax.plot(group.total_payments, group.total_stock_value, marker='o', linestyle='', ms=12, label=name)
	ax.legend(['Is NOT POI','Is POI'])
	plt.xlabel("total_payments")
	plt.ylabel("total_stock_value")
	plt.savefig(imgname)
	
scatter_plot("Outlier1.png")

	
#Drop Outlier 2 - Kenneth Lay and THE TRAVEL AGENCY IN THE PARK
park = "THE TRAVEL AGENCY IN THE PARK"
df_new = df_new[(df_new.index != email) & (df_new.index != park) ]
data_dict.pop( email, None ) 
data_dict.pop( park, None ) 

scatter_plot("Outlier2.png")

#Final table after discarding outliers
df_NaN = df[(df.index != email) & (df.index != "TOTAL") & (df.index != park)].groupby("Names").apply(lambda column: (column == "NaN").sum())
tabulate_data(df_new)	

#Final table with POI
df_new_poi = df_new[df_new["poi"] == 1]
df_NaN = df[(df['poi'] == 1) & (df.index != email) & (df.index != "TOTAL")& (df.index != park)].groupby("Names").apply(lambda column: (column == "NaN").sum())
tabulate_data(df_new_poi)	


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#check for NaN value
def NaNCheck(value):
	if value == 'NaN' or value == 'nan':
		return True

for key, value in my_dataset.items():
	if NaNCheck(my_dataset[key]['total_stock_value']) or NaNCheck(my_dataset[key]['total_payments']) :
		my_dataset[key]['stock_value_ratio'] = 0
	else:
		my_dataset[key]['stock_value_ratio'] = ( float(my_dataset[key]['total_stock_value']) - float(my_dataset[key]['total_payments']))/float(my_dataset[key]['total_payments'])
		

#####################################################################		
####VALIDATING ORIGINAL DATASET							  ###########
#####################################################################
#####################################################################


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Use same cv as tester.py
cv = StratifiedShuffleSplit(100, random_state = 42)

#### FEATURE SELECTION
#Compare reduction techniques between KBest and PCA 
#Compare algorithms at the same time between GaussianNB and DecisionTreeClassifier

# Select parameters and algorithms to compare
tree_dcf = tree.DecisionTreeClassifier()
reduce_dim = [PCA(random_state = 42), SelectKBest(), SelectPercentile()]
classifiers = [GaussianNB(), LinearSVC(random_state = 42), KNeighborsClassifier(), SVC(random_state = 42), tree_dcf]

pipe = Pipeline(steps=[
    ('reduce_dim', PCA()),
    ('classify', LinearSVC())
])
param_grid = [{
		'reduce_dim': reduce_dim,
		'classify': classifiers
    }
	]

def CompareModels(pipe, param_grid, features, labels, features_list):
	#Compare feature selection with Grid SearchCV
	grid = GridSearchCV(pipe, cv=cv, param_grid=param_grid, scoring='f1')
	grid.fit(features, labels)
	
	#Validate classifier with the best settings
	clf = grid.best_estimator_
	test_classifier(clf, my_dataset, features_list)
	return clf

CompareModels(pipe, param_grid, features, labels, features_list)

#####################################################################		
####VALIDATING NEW FEATURE 								  ###########
#####################################################################
#####################################################################

#add new feature to list
features_list_new = features_list
features_list_new.append('stock_value_ratio')	
data2 = featureFormat(my_dataset, features_list_new, sort_keys = True)
labels_new, features_new = targetFeatureSplit(data2)

#CompareModels(pipe, param_grid, features_new, labels_new, features_list_new)

#####################################################################		
#### FEATURE SCALING									  ###########
#####################################################################
#####################################################################


pipe = Pipeline(steps=[
	('scalers', MinMaxScaler()),
    ('reduce_dim', PCA()),
    ('classify', LinearSVC())
])
#CompareModels(pipe, param_grid, features, labels, features_list)


#####################################################################		
#### TUNING PARAMETERS  								  ###########
#####################################################################
#####################################################################

#Restate all variables
pipe = Pipeline(steps=[
    ('reduce_dim', SelectKBest()),
    ('classify', GaussianNB())
])

# Set parameters to compare
param_grid = [{
		'reduce_dim__score_func': [f_classif,  mutual_info_classif],
		'reduce_dim__k': range(1,(len(features_list) - 2))
    }
	]
clf = CompareModels(pipe, param_grid, features, labels, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)