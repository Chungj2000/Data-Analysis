"""
Utilities for correlation related functions.
"""

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap

#Global variables.

gradient = LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)           #Colour gradient of the correlation to turn a matrix into a heatmap.

#Functions

"""
Calculate p-values
"""

#Calculate the p-values of each correlation within a correlation matrix
def Corr_Calculate_pValues(param_corr_data, param_original_data):

    #Calculate statistical significance of correlations between features.
    #Create an empty matrix to store p-values for each correlation during significance testing.
    corr_pValues = pd.DataFrame(
        np.ones(param_corr_data.shape),
        columns=param_corr_data.columns,
        index=param_corr_data.index
    )


    #Iterate through each column pair.
    for column_1 in param_corr_data.columns:
        for column_2 in param_corr_data.columns:

            #Calculate the p-value of the current column pair.
            _, p = pearsonr(
                param_original_data[column_1],
                param_original_data[column_2]
            )

            #Assign the p-value to the newly created data frame.
            corr_pValues.loc[column_1, column_2] = p

    return corr_pValues

"""
p-value mask
"""

#Create a mask that conceals correlations that are not statistically significant.
def Corr_Create_Significance_Mask(param_corr_pValues, param_p_value = 0.05):
    return param_corr_pValues >= param_p_value  

"""
Inverted triangle mask
"""

#Create a mask that conceals the upper triangle in a correlation matrix as well as diagonal correlations.
def Corr_Create_Triangle_Mask(param_corr_data):
    return np.triu(np.ones_like(param_corr_data, dtype=bool))  

"""
Inverted triangle mask
"""

#Plot correlation heatmap.
def Plot_Correlation_Heatmap(param_data, param_mask=None, param_size=(15, 15)):
    plt.figure(figsize=param_size)

    #If mask is on, then only correlations that are statistically significant are shown.
    sns.heatmap(param_data, mask=param_mask, cmap=gradient, vmin=-1, vmax=1, 
                linewidths=0.2, linecolor='black', annot=True, annot_kws={'fontweight':'bold'}, square=True)

    plt.title("Correlation Heatmap of Features")
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()