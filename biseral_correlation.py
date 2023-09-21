# point biserial correlation coefficient

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr

def point_biserial_correlation_column_wise(df: pd.DataFrame, cont: list, cat: list):
    n = len(cont)
    m = len(cat)

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