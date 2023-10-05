import warnings
warnings.simplefilter('always', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import dataframe_image as dfi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import ConfusionMatrixDisplay

# Local imports
import graphviz
from helpers import outliers_IQR, outliers_z_score, outliers_min_max, handle_outliers, fix_obesity
from helpers import model_summary, combined_outliers, plot_pearsonsr_column_wise, plot_chi_square_p_values, plot_point_biserial_correlation
from helpers import BMI 
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import cross_val_score

np.random.seed(2023)


# #################################################
# ######## 2. DATA ANALYSIS AND PROCESSING ########
# #################################################

diabetes = pd.read_csv('diabetes.csv')

binary_features = ['Obesity', 'TCep', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
                   'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                   'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia']
cat_features = ['Race',	'Occupation',	'GP']
num_features = ['Age',	'Height',	'Weight',	'Temperature',	'Urination']
one_hot_features = {} # for future use


# #######################################
# ######## INITIAL DATA ANALYSIS ########
# #######################################

"""
# Code used for answering questions in this section
"""

print('\nINITIAL DATA ANALYSIS')
# number of entries
print("Rows: ", diabetes.shape[0], ", Columns:", diabetes.shape[1])
print("---------------------------------------------------")

# What type of values do you have?
diabetes.info()
print("---------------------------------------------------")

# Duplicates
print("Number of duplicates:", diabetes[diabetes.duplicated(keep=False)].shape[0]/2)
print("---------------------------------------------------")

"""
# ### Interesting findings - Bias in the data set

# There are some interesting things to see in the categoricals ...
# Only one Black in entire data set ...
# This looks very much like a rich person data set ...
"""
print('\nHighlight bias in dataset - skewed proportion in Race and Occupation')
for c in ['Race', 'Occupation']:
  print(diabetes.groupby(c, dropna=False).size())

# Race
#https://en.wikipedia.org/wiki/United_States
demo_race = pd.Series([.62, .12, .06, 2], index=['White', 'Black', 'Asian', 'Mixed/Other'])
diabetes_race = pd.DataFrame(diabetes.groupby('Race').size(), columns = ['Dataset'])/diabetes.shape[0]
diabetes_race = diabetes_race.join(pd.DataFrame(demo_race, columns = ['Population']))
diabetes_race.plot.bar()
plt.title("Race")
plt.savefig("images/diabetes_race.png")

# Gender
#demo_gender = pd.Series([.5, .5], index=['Male', 'Female'])
#diabetes_gender = pd.DataFrame(diabetes.groupby('Gender').size(), columns = ['Dataset'])/diabetes.shape[0]
#diabetes_gender = diabetes_gender.join(pd.DataFrame(demo_gender, columns = ['Population']))
#diabetes_gender.plot.bar()
#plt.title("Gender")
#plt.savefig("images/diabetes_gender.png")

# Income
income_classes = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
pop_income = pd.DataFrame(pd.Series([0.1, 0.2, 0.3, 0.2, 0.15, 0.05, 0, 0, 0, 0, 0, 0], index=income_classes), columns = ['Population'])
ds_income = pd.DataFrame(pd.Series([0, 0, 0, 0.12, 0.2, 0.22, 0.22, 0.16, 0.08, 0, 0, 0], index=income_classes), columns = ['Dataset'])
ds_income = ds_income.join(pop_income) 
ds_income.plot.bar()
plt.title("Income")
plt.savefig("images/diabetes_income.png")

# #####################################
# ######## MISSING DATA PART 1 ########
# #####################################

print('\nMISSING DATA')

missing_data_stats = pd.DataFrame(diabetes.isna().sum().sort_values(ascending=False), columns=['Count'])
missing_data_stats2 = pd.DataFrame(diabetes.isna().mean().sort_values(ascending=False)*100, columns=['Percentage'])
missing_data_stats = missing_data_stats.join(missing_data_stats2)

print("Percentage of missing data:", diabetes.isna().mean().mean())
print("Samples with at least one missing value:", len(diabetes[diabetes.isnull().any(axis=1)]))

# How many values are the samples missing
nomis = diabetes[diabetes.isna().sum(axis=1) == 1].shape[0]
print(f"Percentage of samples with 1 missing values: {nomis/len(diabetes)}")
nomis = diabetes[diabetes.isna().sum(axis=1) > 1].shape[0]
print(f"Percentage of samples with 2 or more missing values: {nomis/len(diabetes)}")

try:
    dfi.export(missing_data_stats.round({'Count': 0, 'Percentage': 2}), 'images/missing_data.png')
except OSError:
    pass

# ### Data Cleaning
# #### Uniform formatting

# Converts all binary features to lower case
for f in binary_features:
   diabetes[f] = diabetes[f].str.lower()

# Converts all binary features to ints, preserving Na-s
diabetes = diabetes.replace({'yes':1, 'no':0}) 
diabetes = diabetes.replace({'Positive':1, 'Negative':0})

# #### Duplicates

# We delete them. We assume they are caused by an error in the data collection, and it's unlikely that there are two correct instances with the exact same values.  
diabetes = diabetes.drop_duplicates(keep='first')

# #### Meters to centimeters
condition = diabetes['Height'] < 100
diabetes.loc[condition, ['Height']] = diabetes.loc[condition, ['Height']].mul(100)

# #### Dropping females and Non-Whites
# Note missing values will also be dropped
diabetes = diabetes[diabetes['Gender'] == 'Male']
diabetes = diabetes[diabetes['Race'] == 'White']


# ### Missing categorical data
# If we don't fill missing categorical data now, we run the risk that either the train or the test set don't contain any NA-s. This can cause a difference in columns after one hot encoding and lead to a crash.
# we want to fill early s.t. there are no Na-s beyond this point
diabetes[cat_features + ['Gender']] = diabetes[cat_features+ ['Gender']].fillna('MISSING')

# ## Train - Test split
# We split as early as possible to avoid cross contamination of information from the test set.
# ### Splitting

train_proportion = 0.8
train_idx = np.random.choice(diabetes.index, int(train_proportion*len(diabetes.index)), replace=False)
train = diabetes.loc[train_idx]
test = diabetes.drop(train_idx)
assert len(diabetes.index) == len(train.index) + len(test.index)

# ##########################
# ######## OUTLIERS ########
# ##########################

# ### Univariate
print('\nOUTLIERS')
try:
    dfi.export(train[num_features].describe().loc[['min','max']], 'images/outliers_1.png')
except OSError:
    pass

# #### Identify Thresholds

# Init with domain knowledge
train_outlier_bounds = pd.DataFrame(
    {'Lower': [16, 110, 30, np.nan, np.nan],
     'Upper': [120, 240, 200, np.nan, np.nan]},
     index=num_features
)

# Statistical
for f in ['Temperature', 'Urination']:
    l_IQR, u_IQR = outliers_IQR(train, f)
    l_Z, u_Z = outliers_z_score(train, f)
    train_outlier_bounds.loc[f] = [max(min(l_IQR, l_Z),0), max(u_IQR, u_Z)]

try:
    dfi.export(train_outlier_bounds.round(2), 'images/outliers_bounds.png')
except OSError:
    pass

# ##### Extent in our data set
outliers_2 = pd.concat(
    [outliers_min_max(train, 'Age', train_outlier_bounds.loc['Age', 'Lower'], train_outlier_bounds.loc['Age', 'Upper']),
    outliers_min_max(train, 'Urination', train_outlier_bounds.loc['Urination', 'Lower'], train_outlier_bounds.loc['Urination', 'Upper'])]
)

try:
    dfi.export(outliers_2[['Age', 'Height', 'Weight', 'Urination', 'Temperature']], 'images/outliers_2.png')
except OSError:
    pass


# box plots for Age and Urination
for f in ['Age', 'Urination']:
    plt.clf()
    bp = train.boxplot(f)
    bp.plot()
    plt.savefig('images/bp_'+f+'.png') 

# #### Deal with

train = handle_outliers(train, train_outlier_bounds)
test = handle_outliers(test, train_outlier_bounds)

try:
    # How does it look now? All min max values sensible ...
    dfi.export(train[num_features].describe().loc[['min','max']], 'images/outliers_minmax_2.png') 
except OSError:
    pass

for f in ['Age', 'Urination']:
    plt.clf()
    bp = train.boxplot(f)
    bp.plot()
    plt.savefig('images/bp_'+f+'_after.png') 

# ### Combined outliers
# Combined outliers must be handled after fixing the individual ones, otherwise the same ones would be discovered
# Just use formula from lecture stating that x should be standardized.

print(train[num_features])

zs_train = combined_outliers(train[num_features], num_features)

plt.clf()
plt.figure()
plt.scatter(range(0, len(zs_train)), sorted(zs_train))
plt.savefig('images/combined_scatter.png') 
plt.clf()

print(train[zs_train > 3][['Age', 'Gender', 'Race', 'Occupation', 'Height', 'Weight', 'Urination', 'Temperature']])

try:
    dfi.export(train[zs_train > 3][['Age', 'Gender', 'Race', 'Occupation', 'Height', 'Weight', 'Urination', 'Temperature']], 'images/combined_outlier.png')
except OSError:
    pass

# #### Handle
train = train[zs_train < 4]
zs_test = combined_outliers(test, num_features, test.copy())
test = test[zs_test < 4]

# ## Missing data - Part 2

# ### Deal with

# #### Derived Features

train = fix_obesity(train)
test = fix_obesity(test)

# #### Missing Binaries

train[binary_features] = train[binary_features].fillna(0)
test[binary_features] = test[binary_features].fillna(0)

# #### Missing numeric

# Fill Na-s with mean. 
train[num_features] = train[num_features].fillna(train[num_features].mean())
# We fill the test data with the mean of the train data, making the test set indepentent of each others
test[num_features] = test[num_features].fillna(train[num_features].mean())

# #### Testing for remaining Na-s

assert train.isna().sum().sum() == 0, f'train data still containts {train.isna().sum().sum()} Na-s'
assert test.isna().sum().sum() == 0, f'test data still containts {test.isna().sum().sum()} Na-s'

# Henceforth, we may assume both train and test data contains no Na-s, drastically simplifying the rest of the code. For online learning, we should perhaps implement the code in this section as a function so that we can apply it on new cases continuously.

# ## Encoding

gender_dummies_train = pd.get_dummies(train['Gender'], prefix='gender')
train = train.join(gender_dummies_train)

gender_dummies_test = pd.get_dummies(test['Gender'], prefix='gender')

test = test.join(gender_dummies_test)

one_hot_features['Gender'] = list(set(list(gender_dummies_train.columns) + list(gender_dummies_test.columns)))

# Add one hot variable column to train/test if it exists for one, but not the other adn fill with 0
for i in list(gender_dummies_train):
    if i not in gender_dummies_test:
        test[i] = 0

for i in list(gender_dummies_test):
    if i not in gender_dummies_train:
        train[i] = 0

def checkEqualColumns(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

assert checkEqualColumns(train.columns, test.columns), f'train and test set doesent have the same features'

for i in zip(sorted(train.columns), sorted(test.columns)):
    assert i[0] == i[1], 'Train and test columns do not match'


train['BMI'] = BMI(train['Weight'],  train['Height'])
test['BMI'] = BMI(test['Weight'],  test['Height'])


assert len(train.index) > 10

# ##############################
# ######## CORRELATIONS ########
# ##############################

corr = train.corr(numeric_only=True)
#corr['Diabetes']
#sns.heatmap(corr)

print('\nCORRELATIONS')
# look at the smallest and largest in absolute value

corrs = corr.stack().loc[lambda x : (x < 1)].abs().sort_values()
print("Smallest:")
print(corrs[:20])
print("-------------------------------")
print("Largest:")
print(corrs[-20:])
#corr.style.background_gradient(cmap=cmap).set_precision(2)

cmap = 'coolwarm' # Added colour map as a variable for consistent plot style
plot_pearsonsr_column_wise(train[num_features + ['BMI']],kwargs={'cmap' : cmap, 'center':0}, outfile='images/cont_cont_corr.png')
# reverse color as low p-value indicates strong dependence
plot_chi_square_p_values(train[binary_features  + cat_features +  ['Diabetes']], kwargs={'cmap' : matplotlib.colormaps[cmap +'_r']}, outfile = 'images/cat_cat_corr.png')
plot_point_biserial_correlation(train, cont=num_features + ['BMI'], cat=binary_features + ['Diabetes'], kwargs={'cmap' : cmap}, outfile = 'images/cont_cat_corr_bmi.png')

#  We see that weight and obesity is strongly correlated, however BMI and obesity is not. Furthermore, diabetes has no correlation with either of them. This does not mean that BMI or weight are bad predictors, since the relationship between them could be non-linear. 
#  Urination is indeed very correlated, which is apparent in the later plots.

# ## Data exploration

# NOT USED REMOVE

train_plot = train[num_features + ['BMI', 'Diabetes']]

#diabetes_plot = train.drop('Gender', axis=1)
#g = sns.pairplot(train_plot, hue='Diabetes')
#plt.savefig('images/pairplot_diabetes.png')

#g = sns.pairplot(train_plot, kind='reg')


# ###################################
# ######## FEATURE SELECTION ########
# ###################################

print("Temperature has low variance. Coefficient of variation = stdev/mean =", np.std(train['Temperature'])/np.mean(train['Temperature']))

# Some sanity checks
assert train.isna().sum().sum() == 0, 'No Na-s should be present after handling. They must have been introduced'

#selected_features = ['Urination', 'Age'] + binary_features
selected_features = num_features + binary_features 
selected_features.remove('Urination')
selected_features.remove('Temperature')
selected_features.remove('Obesity')
selected_features.remove('TCep')

print(f'\nSELECTED FEATURES')
print('count:',len(selected_features))
print(selected_features)

# ## Training model

X_train = train[selected_features]
y_train = train['Diabetes']

# Test set
X_test = test[selected_features]
y_test = test['Diabetes']


for index in X_train.dtypes.keys():
    dtype = X_train.dtypes[index]
    assert dtype == 'float64' or dtype == 'int64' or dtype == 'uint8', f"feature '{index}' is not of type float or int but {dtype}"


print('Number of rows:', len(X_train.index))
print('Number of features:', len(X_train.columns))
print(X_train.columns)

# ###################################
# ######## DECISION TREE ############
# ###################################

# ## Baseline classifier
# A simple Naive Bayes classifier to witch we compare our results. (ours are barely better)

# Not used for anything , remove
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ## Decision Tree cross validation

depths = list(range(1, 20))
scores = {}

for d in depths:
    clf_cv = tree.DecisionTreeClassifier(max_depth=d)
    score = cross_val_score(clf_cv, X_train, y_train, cv=5).mean()
    scores[d] = score
    #print(f'Decision tree with depth={d}. cv score: {score}') # printing for debugging

clf = tree.DecisionTreeClassifier(max_depth=7)
clf_full_tree = clf.fit(X_train, y_train)

y_test_pred = clf_full_tree.predict(X_test)

confusion_mat = metrics.confusion_matrix(y_test, y_test_pred)
con_mat_disp = ConfusionMatrixDisplay(confusion_mat, display_labels=clf.classes_)
con_mat_disp.plot()
plt.clf()

# ### Some Pruning
# Code borrowed from https://www.kaggle.com/code/arunmohan003/pruning-decision-trees-tutorial

path = clf_full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
# For each alpha we will append our model to a list
clfs = []
for ccp_alpha in ccp_alphas:
    clf_tmp = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_tmp.fit(X_train, y_train)
    clfs.append(clf_tmp)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas, node_counts)
plt.scatter(ccp_alphas, depth)
plt.plot(ccp_alphas, node_counts,label='no of nodes', drawstyle="steps-post")
plt.plot(ccp_alphas, depth,label='depth', drawstyle="steps-post")
plt.legend()
plt.savefig('images/pruned_tree_complexity.png')


# Alpha 0.02 seems like a good trade off between size and complexity
# as indicated by pruned_tree_complexity.png
alpha = 0.02
classes = ['Negative', 'Positive']
clf_pruned_tree = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
clf_pruned_tree.fit(X_train,y_train)
y_train_pred = clf_pruned_tree.predict(X_train)
y_test_pred = clf_pruned_tree.predict(X_test)

print('RESULTS')
print('\nPruned tree')
#print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

plt.figure(figsize=(20, 20))
tree.plot_tree(clf_pruned_tree, feature_names=selected_features, class_names=classes, filled=True)
plt.savefig('images/pruned_tree.png')