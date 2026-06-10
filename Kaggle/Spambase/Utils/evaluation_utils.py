"""
Utilities for model evaluation related functions.
"""

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#Global variables.



#Functions

"""
Evaluate Model - K-fold Cross Validation
"""

#Train the model for each k-fold and evaluate the model performance based on training data set used.
def Evaluate_kFold_Scores(param_validator, param_model, param_X_Train, param_Y_Train):
    return cross_val_score(param_model, param_X_Train, param_Y_Train, cv=param_validator, scoring='accuracy')


"""
Evaluate Model - Accuracy Score
"""

#Calculate accuracy score of the model.
def Evaluate_Accuracy_Score(param_y_test, param_predict):    
    print(f'Accuracy Score: {accuracy_score(param_y_test, param_predict)}')

"""
Evaluate Model - F1-Score
"""

#Calculate f1-score for the model at classifying spam and ham.
def Evaluate_F1_Score(param_y_test, param_predict, param_labels):
    f1 = f1_score(param_y_test, param_predict, average=None)
    print(f'F1 Score ({param_labels[0]}): {f1[0]}')
    print(f'F1 Score ({param_labels[1]}): {f1[1]}')

"""
Evaluate Model - Confusion Matrix
"""

#Generate a confusion matrix showing true and false positives/negatives of the model.
def Evaluate_Confusion(param_y_test, param_predict, param_labels):
    model_confusion = confusion_matrix(param_y_test, param_predict)
    #print(confusion_matrix(param_test, param_predict))

    plt.figure(figsize=(6,6))
    sns.heatmap(model_confusion, cmap='Greens', square=True, annot=True, annot_kws={'fontweight' : 'bold'}, fmt='g', xticklabels=param_labels, yticklabels=param_labels)
    plt.title('Confusion Matrix for Model')
    plt.xlabel('Predicted Placement')
    plt.ylabel('True Placement')
    plt.show()