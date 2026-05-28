"""
Utilities for correlation related functions.
"""

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

#Global variables.



#Functions

"""
Create Classification Model - Random Forest
"""

#Reusable function for creating new random forest models for each unique use case.
def Create_RandomForest_Model(param_Depth, param_X, param_Y):
    randomForest_Model = RandomForestClassifier(max_depth=param_Depth, random_state=1)                    #Define depth of model.
    randomForest_Model.fit(param_X, param_Y)                                                              #Build forest model using data.
    return randomForest_Model

"""
Random Forest - Feature Importances
"""

#List and sort the estimated importaance of each features within the model.
def Get_RandomForest_Importances(param_Model, param_Features):
    
    #Create a dataframe to contain a list of features and their corresponding importance in the model.
    feature_Importances = pd.DataFrame({
        'feature': param_Features,
        'importance': param_Model.feature_importances_
    })

    #Sort the features based on the importance value.
    feature_Importances = feature_Importances.sort_values(by='importance', ascending=False).reset_index(drop=True)

    return feature_Importances