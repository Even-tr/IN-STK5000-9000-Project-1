import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr
from sklearn.metrics import accuracy_score,confusion_matrix, precision_recall_fscore_support
from scipy.stats import pointbiserialr


def pearsonr_column_wise(df: pd.DataFrame):
    n = len(df.columns)
    colnames = df.columns

    res = np.zeros(shape=(n, n)) # to store results
    for i in range(n):
        for j in range(n):


            # Crosstab constructs the Contingency table for column i against j
            test_result = pearsonr(df[colnames[i]], df[colnames[j]])
            #print(test_result.pvalue)
            res[i,j] = test_result.statistic

    return res

def chi_square_column_wise(df: pd.DataFrame):
    n = len(df.columns)
    colnames = df.columns

    res = np.zeros(shape=(n, n)) # to store results
    for i in range(n):
        for j in range(n):


            # Crosstab constructs the Contingency table for column i against j
            test_result = chi2_contingency(pd.crosstab(df[colnames[i]],
                                                       df[colnames[j]]).to_numpy())
            #print(test_result.pvalue)
            res[i,j] = (test_result.pvalue > 0.05)*1
    return res

def plot_pearsonsr_column_wise(df: pd.DataFrame, outfile=None, kwargs={'cmap':'crest'}):
    """
    plots a correlation matrix for categorical columns using Chi-square score.

    parameters
    ----------
    df : pd.DataFrame
        data to check for correlations
    outfile : str, optional
        location for saving figure
    kwargs: dict, optional
        key word arguments to pass to the sns plot
    """
    plt.clf()
    arr = pearsonr_column_wise(df)
    ax = sns.heatmap(arr, xticklabels = df.columns, yticklabels = df.columns, **kwargs)
    ax.tick_params(top=False, labeltop=True, bottom=False, labelbottom=False, left=False)
    ax.set_aspect('equal')
    plt.title('Pearson\'s r column wise')
    plt.xticks(rotation = 90)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()




def plot_chi_square_p_values(df: pd.DataFrame, outfile=None, kwargs={'cmap':'crest'}):
    """
    plots a correlation matrix for categorical columns using Chi-square score.

    parameters
    ----------
    df : pd.DataFrame
        data to check for correlations
    outfile : str, optional
        location for saving figure
    kwargs: dict, optional
        key word arguments to pass to the sns plot
    """
    plt.clf()
    arr = chi_square_column_wise(df)
    ax = sns.heatmap(arr, xticklabels = df.columns, yticklabels = df.columns, cbar=False, **kwargs)
    ax.tick_params(top=False, labeltop=True, bottom=False, labelbottom=False, left=False)
    ax.set_aspect('equal')
    plt.title('p < 0.05 for column wise Chi-square test of independence')
    plt.xticks(rotation = 90)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

# Modified to return min, max values so it can be used on test set
def outliers_IQR(df, feature):
  try:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1

    # lower bounds
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return lower, upper
    #outliers = df[(df[feature] < lower) | (df[feature] > upper)]
    #return outliers
  except Exception as e:
    print("invalid feature")

def outliers_z_score(df, feature, no_z=3):
  # returns blabla
  lower = df[feature].mean()-no_z*df[feature].std()
  upper = df[feature].mean()+no_z*df[feature].std()
  return lower, upper

# Generalize min max rule used for age
# Returns DF of outliers, maybe use index instead?
def outliers_min_max(df, feature, min=None, max=None):
  try:
    cond_min = df[feature] < min if min != None else False
    cond_max = df[feature] > max if max != None else False
    return df[cond_min | cond_max ]
  except Exception as e:
    print("invalid feature")
    
# Function that can be run on both training set and test set
def handle_outliers(df, df_bounds):
  for f in df_bounds.index:
      outliers = outliers_min_max(df, f,
                                  min=df_bounds.loc[f, 'Lower'],
                                  max=df_bounds.loc[f, 'Upper']
                                )
      df.loc[outliers.index, f] = np.NaN
  return df

def BMI(weight, height):
  return weight/(height**2/(100*100))

def fix_obesity(df, threshold=30):
  idx = df[df['Obesity'].isna()].index
  # This is ugly ...
  idx2 = df.loc[idx,].loc[BMI(df.loc[idx,]["Weight"], df.loc[idx,]["Height"]) <= threshold].index
  # maybe not set obesity to 1 for high BMI, to avoid "Body Builder" problem
  # It is possible to have high BMI without Obesity, the case of low BMI and Obesity harder to imagine
  idx3 = df.loc[idx,].loc[BMI(df.loc[idx,]["Weight"], df.loc[idx,]["Height"]) > threshold].index
  #print(idx2)
  #print(idx3)
  df.loc[idx2,'Obesity'] = 0
  df.loc[idx3,'Obesity'] = 1
  #df.loc[idx,]
  return df

def model_summary(clf, X_test, y_test, header = True, name=''):
   # computes the accuracy, precision and recall for a classifier and prints a standard output.
   assert 'predict' in dir(clf), "Classifier must have a 'predict' method"

   y_pred = clf.predict(X_test)

   acc = accuracy_score(y_test, y_pred)
   prec, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
   if header:
      print('Accuracy\tPrecision\tRecall')
   print(f'{acc :.2f}\t\t{prec:.2f}\t\t{recall:.2f}\t{name}')


#def combined_outliers(df: pd.DataFrame, features: list):
#   # Calculate combined outliers for continous features using euclidean norm
#  assert len(features) > 1
#  df = df[features].fillna(df.mean())
#  #assert df.isna().sum().sum() == 0, 'No Na-s must be present'
#  df = df.to_numpy()
#  df = (df - df.mean())/(df.std()) # normalize to be indepentent of parameterization
#  d = np.sqrt(np.square(df).sum(axis=1)) #Calculate square distance 
#  z = (d-d.mean())/d.std()   # Normalize distances
#  return z

def combined_outliers(train: pd.DataFrame, features: list, test: pd.DataFrame = None):
   # Calculate combined outliers for continous features using euclidean norm
  assert len(features) > 1
  #train  = train.copy()
  train = train[features]
  train.fillna(train.mean())
  train = train.to_numpy()
  train = (train - train.mean())/(train.std()) # normalize to be indepentent of parameterization
  d_train = np.sqrt(np.square(train).sum(axis=1)) #Calculate square distance
  z_train = (d_train-d_train.mean())/d_train.std()   # Normalize distances
  ret = z_train
  # If we get a test set we need to use parameters from the training set
  if test is not None:
     test = test[features].fillna(train.mean())
     test = test.to_numpy()
     test = (test - train.mean())/(train.std()) # normalize to be indepentent of parameterization
     d_test = np.sqrt(np.square(test).sum(axis=1)) #Calculate square distance
     z_test = (d_test-d_test.mean())/d_test.std()   # Normalize distances
     ret = z_test
  return ret



def point_biserial_correlation_column_wise(df: pd.DataFrame, cont: list, cat: list):
    n = len(cont)
    m = len(cat)
    df = df[cont + cat].astype('float64')

    res = np.zeros(shape=(n, m)) # to store results
    for i in range(n):
        for j in range(m):


            # Crosstab constructs the Contingency table for column i against j
            test_result = pointbiserialr(df[cont[i]], df[cat[j]])
            #print(test_result.pvalue)
            res[i,j] = test_result.statistic
    return res


def plot_point_biserial_correlation(df: pd.DataFrame, cont: list, cat: list, outfile=None, kwargs={'cmap':'crest'}):
    """
    plots a correlation matrix for categorical columns using Chi-square score.

    parameters
    ----------
    df : pd.DataFrame
        data to check for correlations
    outfile : str, optional
        location for saving figure
    kwargs: dict, optional
        key word arguments to pass to the sns plot
    """
    plt.clf()
    arr = point_biserial_correlation_column_wise(df, cont, cat)
    ax = sns.heatmap(arr, xticklabels = cat, yticklabels = cont, **kwargs)
    ax.tick_params(top=False, labeltop=True, bottom=False, labelbottom=False, left=False)
    ax.set_aspect('equal')
    plt.title('Point biserial correlation coefficient')
    plt.xticks(rotation = 90)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()