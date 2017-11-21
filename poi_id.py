#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', 'total_payments', 'total_stock_value'] # You will need to use more features

features_list = ['poi', 'total_payments', 'total_stock_value', 'salary', 'deferral_payments', 'exercised_stock_options',
                     'bonus', 'restricted_stock', 'restricted_stock_deferred',
                     'expenses', 'loan_advances', 'other',  
					 'director_fees', 'deferred_income',
                     'long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

	
#DATA EXPLORATION	
import pandas as pd

df = pd.DataFrame.from_dict(data_dict, orient='index') 
df.index.name = "Names"
df.sort_index(axis=1, inplace=True)

print("No. of Data Points: ", len(df))	
print("No. of Features: ",len(df.columns))
print("No. of POI: ", df["poi"].sum())

print("##################")
df2 = df.replace("NaN", 0)

from tabulate import tabulate
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

#Drop Outlier 1
df_new = df_new[df_new.index != email]
data_dict.pop( "TOTAL", None ) 

df_NaN = df[df.index != email].groupby("Names").apply(lambda column: (column == "NaN").sum())
tabulate_data(df_new)	

#CREATE SCATTER PLOT
email = df_new[df_new['total_payments'] == df_new['total_payments'].max()].index[0]
print("Email Address with Max Value #2 :", email)

import matplotlib
import matplotlib.pyplot as plt


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
	plt.show()
	
scatter_plot("Outlier1.png")

	
#Drop Outlier 2
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
		

# Add new feature to feature_list
features_list.append('stock_value_ratio')	
print(features_list)													
														
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#split data for testing
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#scale data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_scaled_train = scaler.fit_transform(features_train)
features_scaled_test = scaler.fit_transform(features_test)


#### FEATURE SELECTION
#Use KBest to select best features (http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

kbest = SelectKBest(chi2, k=2)
X_train_kbest = kbest.fit_transform(features_scaled_train, labels_train)
X_test_kbest= kbest.fit_transform(features_scaled_test, labels_test)

print("KBest Feature Selection: ",kbest.get_support())
print("KBest Scores: ",kbest.scores_)

#Use PCA as comparison
from sklearn.decomposition import PCA
#n_components=
pca = PCA()
X_train_pca = pca.fit_transform(features_scaled_train)
X_test_pca = pca.fit_transform(features_scaled_test)
print("PCA variance ratio:", pca.explained_variance_ratio_ )
print("PCA Score:", pca.score(features_scaled_train))

#Compare reduction techniques http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

#use Pipeline 
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', LinearSVC())
])

N_FEATURES_OPTIONS = [2, 4, 8, 10]
C_OPTIONS = [1, 10, 100, 1000]

param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7)],
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
clf = GridSearchCV(pipe, cv=2, n_jobs=1, param_grid=param_grid)
clf.fit(features_scaled_train, labels_train)
mean_scores = np.array(clf.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)


plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
	rects = plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])
	
	for i, rect in enumerate(rects):
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width()/2., 1.025*height,
			'%.2f' % reducer_scores[i],
			ha='center', va='bottom')
	
plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 4, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.yticks(np.arange(0, 1, 0.1))
plt.legend(loc='lower right')
plt.savefig("FeatureSelection.png")
plt.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

#Get Initial score

pred = clf.predict(features_scaled_test)
score = clf.score(features_scaled_test, labels_test)
print("Initial Accuracy Score: ", score)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)