#!/usr/bin/python

#Import all libraries
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from tabulate import tabulate
import matplotlib.pyplot as plt

#Import all sklearn libraries
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from time import time


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

df_NaN = df[df.index != email].groupby("Names").apply(lambda column: (column == "NaN").sum())
tabulate_data(df_new)	

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

	
#Drop Outlier 2 - Kenneth Lay
df_new = df_new[df_new.index != email]
data_dict.pop( email, None ) 

scatter_plot("Outlier2.png")

#Final table after discarding outliers
df_NaN = df[(df.index != email) & (df.index != "TOTAL")].groupby("Names").apply(lambda column: (column == "NaN").sum())
tabulate_data(df_new)	

#Final table with POI
df_new_poi = df_new[df_new["poi"] == 1]
df_NaN = df[(df['poi'] == 1) & (df.index != email) & (df.index != "TOTAL")].groupby("Names").apply(lambda column: (column == "NaN").sum())
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
		

		
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#scale data using MinMaxScaler

scaler = MinMaxScaler()
features_scaled= scaler.fit_transform(features)


#### FEATURE SELECTION
#Compare reduction techniques between KBest and PCA http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py

#use Pipeline 
pipe = Pipeline(memory=None, steps=[
    ('reduce_dim', PCA()),
    ('classify', LinearSVC())
])

N_FEATURES_OPTIONS = [2, 4, 8, 10]
C_OPTIONS = [1, 10, 100, 1000]

param_grid = [
    {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA', 'KBest(chi2)']
cv = StratifiedShuffleSplit(n_splits=100, random_state= 42)

#Compare feature selection with Grid SearchCV and display chart
grid = GridSearchCV(pipe, cv=cv, n_jobs=1, param_grid=param_grid)
grid.fit(features_scaled, labels)
print("Grid Search CV Best Params: ",grid.best_params_)

#Generate mean scores
mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='lower right')
plt.savefig("Feature_Selection.png")


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

tree_dcf = tree.DecisionTreeClassifier(min_samples_split=2, random_state = 42)

pipe_NB = Pipeline(memory=None, steps=[
    ('PCA', PCA(n_components=10)),
    ('classify_NB', GaussianNB())
])

pipe_tree = Pipeline(memory=None, steps=[
    ('PCA', PCA(n_components=10)),
    ('classify_tree', tree_dcf)
])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

features_scaled_train = scaler.fit_transform(features_train)
features_scaled_test = scaler.fit_transform(features_test)

### Evaluate effects of scaling with PCA
#PCA before scaling
pca = PCA(n_components=10)
pca.fit(features_train, labels_train)
pca.transform(features_test)

# summarize components
print("Explained Variance: ", pca.explained_variance_ratio_)  

#PCA after scaling
pca_new = PCA(n_components=10)
pca_new.fit(features_scaled_train, labels_train)
pca_new.transform(features_scaled_test)

# summarize components
print("Explained Variance (with scaled features): ", pca_new.explained_variance_ratio_)  
print("##########################")

### TEST NB SCORE WITH SCALED AND UNSCALED FEATURES
pipe_NB_test = pipe_NB.fit(features_train, labels_train)
pred_test = pipe_NB_test.predict(features_test)
print("GNB Score with Unscaled Features: ", accuracy_score(pred_test, labels_test))


pipe_NB.fit(features_scaled_train, labels_train)
pred2 = pipe_NB.predict(features_scaled_test)
print("GNB Score with Scaled Features: ", accuracy_score(pred2, labels_test))
print("##########################")


####### TESTING DECISION TREE NOW
pipe_tree_test = pipe_tree.fit(features_train, labels_train)
pred = pipe_tree_test.predict(features_test)
acc = accuracy_score(pred, labels_test)
print("DecisionTree Score with Unscaled features: ", acc)


pipe_tree.fit(features_scaled_train, labels_train)
pred = pipe_tree.predict(features_scaled_test)
acc = accuracy_score(pred, labels_test)
print("DecisionTree Score with Scaled features: ", acc)
	
print("##########################")
	
####### TUNING TREE	
max_depth = range(1,20)
param_grid = [
	{
	'classify_tree__criterion' : ["gini"],
	'classify_tree__splitter' : ["best", "random"],
	'classify_tree__max_depth':max_depth
	},
	{
	'classify_tree__criterion' : ["entropy"],
	'classify_tree__splitter' : ["best", "random"],
	'classify_tree__max_depth':max_depth
	}
]

clf = GridSearchCV(
	pipe_tree, 
	cv=cv, 
	param_grid=param_grid)
clf.fit(features_scaled_train, labels_train)
clf.predict(features_scaled_test)
print("Grid Search CV Best Params",clf.best_params_)
print("Grid Search CV Score",clf.score(features_scaled_test,labels_test ))


y_true, y_pred = labels_test, clf.predict(features_scaled_test)
print(classification_report(y_true, y_pred))


#Test score with new feature
features_list_new = features_list
features_list_new.append('stock_value_ratio')	
data2 = featureFormat(my_dataset, features_list_new, sort_keys = True)
labels_new, features_new = targetFeatureSplit(data2)

features_train_new, features_test_new, labels_train_new, labels_test_new = \
    train_test_split(features_new, labels_new, test_size=0.3, random_state=42)

features_scaled_train_new = scaler.fit_transform(features_train_new)
features_scaled_test_new = scaler.fit_transform(features_test_new)
							
clf2 = GridSearchCV(
	pipe_tree, 
	cv=cv, 
	param_grid=param_grid)
clf2.fit(features_scaled_train_new, labels_train_new)
clf2.predict(features_scaled_test_new)
print("Grid Search CV Best Params - with new feature: ",clf2.best_params_)
print("Grid Search CV Score - with new feature: ",clf2.score(features_scaled_test_new,labels_test_new ))
y_true_new, y_pred_new = labels_test_new, clf2.predict(features_scaled_test_new)
print(classification_report(y_true_new, y_pred_new))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf2, my_dataset, features_list_new)



