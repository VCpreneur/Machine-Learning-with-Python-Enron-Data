#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from collections import defaultdict
import operator
from math import isnan
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import main


#---------------------------------------------------------------------#
### Task 1: Select what features you'll use.
#---------------------------------------------------------------------#
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = [
                 'poi',
                 'salary',
                 'bonus',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'other',
                 'deferred_income',
                 'long_term_incentive',
                 'director_fees',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'salary_to_avg_salary', # new feature to be created
                 'bonus_to_avg_bonus', # new feature to be created
                ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Print the length of the dataset:
print '\n Lenght of the dataset: ', len(data_dict)

# Count the number of All NaN values in the whole dataset:
all_values = 0
all_NaNs = 0
for name, features in data_dict.iteritems():
    for k, v in features.iteritems():
        all_values +=1
        if v == 'NaN':
            all_NaNs +=1

print '\n NaN values are {}% of all feature values in the whole dataset!'.\
        format(round(float(all_NaNs)/float(all_values)*100))
print all_values

# Count the number of NaN values for each feature:
def count_NaNs(feature_name):
    feature_values = 0
    feature_NaNs = 0
    for name, features in data_dict.iteritems():
        if features[feature_name] == 'NaN':
            feature_NaNs +=1
        elif features[feature_name] != 'NaN':
            feature_values +=1             
    print '\n {} feature has {}% NaN values.'.\
        format(feature_name, round(float(feature_NaNs)/float(feature_NaNs+feature_values)*100))

for feature in features_list[0:20]:
    count_NaNs(feature)
    
    
# print one of the key/value pairs in the dictionary to learn more about it:
print '\n One entry in the dataset: \n', data_dict.items()[0]

# printing the length of features for evey person:
print '\n Number of features per person: ', len(data_dict.values()[0])

# printing the number of missing Persons of Interest classifications provided:
POIs = 0
nPOIs = 0
for k, v in data_dict.iteritems():
    if v["poi"] == 1:  # (or True)
        POIs += 1
    else: 
        nPOIs += 1
# OR 
#POIs = sum(person['poi'] for person in data_dict.values())
#nPOIs = sum(not person['poi'] for person in data_dict.values())       
print "\n Number of POIs: {}, \n Number of non-POIs: {}, \n Total Provided: {}, \n Missing Classifications: {}".\
        format(POIs, nPOIs, POIs+nPOIs, len(data_dict)-(POIs+nPOIs))



#---------------------------------------------------------------------#
### Task 2: Remove outliers
#---------------------------------------------------------------------#


# Separate a feature in a new dictionary to find it's outliers (repeated for important features):
temp_dict = defaultdict(int)
temp_feature = 'salary'

for name, features in data_dict.iteritems():
    temp_dict[name] = data_dict[name][temp_feature]

temp_dict = {k: temp_dict[k] for k in temp_dict if not isnan(float(temp_dict[k]))}

sorted_temp_dict_by_value = sorted(temp_dict.items(), key=operator.itemgetter(1), reverse=True)
print '\n Sorted {} feature by values (performed for every feature to find outliers): '.format(temp_feature)
print sorted_temp_dict_by_value

#sorted_temp_dict_by_key = sorted(temp_dict.items(), key=operator.itemgetter(0))
#print sorted_temp_dict_by_key
#print '\n'


# Print the entry in the dataset of the name 'Total':
print '\n The TOTAL entry row: \n', data_dict['TOTAL']

# Removing the Total entry:
data_dict.pop("TOTAL")


#---------------------------------------------------------------------#
### Task 3: Create new feature(s)
#---------------------------------------------------------------------#

# Calculat the sum of all salaries:
all_salaries = 0    
for k, v in data_dict.iteritems():
    if v["salary"] != 'NaN':  
        all_salaries = all_salaries + v["salary"]
    else: 
        pass
print '\n All Salaries: ', all_salaries

# Create a new feature 'salary_to_avg_salary':
for name, features in data_dict.iteritems():
    try:
        features['salary_to_avg_salary'] = float(features['salary']) / float(all_salaries)
    except:
        pass
    if isnan(features['salary_to_avg_salary']):
        features['salary_to_avg_salary']= 0

        
# Calculat the sum of all bonuses:
all_bonuses = 0    
for k, v in data_dict.iteritems():
    if v["bonus"] != 'NaN':  
        all_bonuses = all_bonuses + v["bonus"]
    else: 
        pass
print '\n All Bonuses: ', all_bonuses

# Create a new feature 'bonus_to_avg_bonus':
for name, features in data_dict.iteritems():
    try:
        features['bonus_to_avg_bonus'] = float(features['bonus']) / float(all_bonuses)
    except:
        pass
    if isnan(features['bonus_to_avg_bonus']):
        features['bonus_to_avg_bonus']= 0


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#---------------------------------------------------------------------#
### Task 4: Try a varity of classifiers
#---------------------------------------------------------------------#
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# Import needed methods
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


scaler = MinMaxScaler()
NB = GaussianNB()
DT = DecisionTreeClassifier()
SV = SVC()

# Set the parameters for all used algorithms to use in Task 5:
param_grid = {
              'selectkbest__k': range(2,22),
              #'tree__random_state' : [42],
              #'tree__criterion' : ['gini', 'entropy'],
              #'tree__max_depth' : [None, 1, 2, 3, 4],
              #'tree__min_samples_split' : [2, 3, 4, 25],
              #'svm__kernel' : ['rbf'],
              #'svm__C' : [1, 10, 100, 1000, 10000],
              }

# Create the pipline to use in Task 5
pipeline = Pipeline([
                    #('min_max_scaler', scaler),
                    ('selectkbest', SelectKBest()),
                    ('naive_bayes', NB),
                    #('tree', DT),
                    #('svm', SV),
                    ])
                  

#---------------------------------------------------------------------#
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
#---------------------------------------------------------------------#
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#clf = clf.fit(features_train, labels_train)     
#pred = clf.predict(features_test)


grid_search = GridSearchCV(pipeline, cv=5, n_jobs=1, param_grid=param_grid)
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
pred = clf.predict(features_test)

print '\n', clf
print '\n', "Best parameters are: ", grid_search.best_params_, '\n'

selected_features=[features_list[i+1] for i in clf.named_steps['selectkbest'].get_support(indices=True)]
#print '\n Selected Features: ', selected_features
scores = clf.named_steps['selectkbest'].scores_
#print '\n Scores of Selected Features: ', scores

indices = np.argsort(scores)[::-1]
#print indices
print 'The ', len(selected_features), " features selected and their scores:"
for i in range(len(selected_features)):
    #print i
    #print indices[i]
    print "feature no. {}: {} ({})".format(i+1,selected_features[i], scores[indices[i]])


#print "grid search results: \n"
#print grid_search.cv_results_

#---------------------------------------------------------------------#
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. 
#---------------------------------------------------------------------#
### You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


print '\n Udacity Tester results: \n'
main()

