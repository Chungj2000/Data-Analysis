"""
Utilities for correlation related functions.
"""

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Global variables.



#Functions

"""
Distribution plot - KDE
"""

#Reusable Function to plot Log-Transformed KDE plots with multiple features being compared.
def log_kde_plots(param_data, param_features, param_legends, param_labels):

    fig, axes = plt.subplots(1, len(param_features), figsize=(12, 5))

    for i, feature in enumerate(param_features):                                              #Create a new plot for each feature

        for legend in param_legends.unique():                                                 #Create line graphs for each legend, in this case being spam or ham classification.
            subset = param_data[param_legends == legend][feature]                                   #Save data points of the feature of each legend.
            sns.kdeplot(np.log1p(subset), label=param_labels[legend], ax=axes[i])             #Plot the data points as a KDE using log-transformed values.

        axes[i].set_title(feature)
        axes[i].set_xlabel("Log-Transformed Values")
        axes[i].set_ylabel("Log-Transformed Density")
        axes[i].legend()
        
    plt.tight_layout()
    plt.show()