import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency



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
            res[i,j] = test_result.pvalue
    return res



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
    arr = chi_square_column_wise(df)
    ax = sns.heatmap(arr, xticklabels = df.columns, yticklabels = df.columns, **kwargs)
    ax.tick_params(top=False, labeltop=True, bottom=False, labelbottom=False, left=False)
    ax.set_aspect('equal')
    plt.title('P-values for column wise Chi-square test of independence')
    plt.xticks(rotation = 90)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

