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

#Train the model for each k-fold and evaluate the model performance based on training data set used, and return an array of scores for each split.
def Evaluate_kFold_Scores(param_validator, param_model, param_x_train, param_y_train):
    return cross_val_score(param_model, param_x_train, param_y_train, cv=param_validator, scoring='accuracy')



"""
Evaluate Model - Describe K-Fold Scores
"""

#List out the scores for each k-fold, as well as mean and standard deviation.
def Describe_kFold_Scores(param_scores):
    for k, score in enumerate(param_scores, start=1):
        print(f'Score of Split {k}: {score}')

    print("")

    print(f'Average Score per K-fold: {np.mean(param_scores)}')
    print(f'Standard Deviation of Scores: {np.std(param_scores)}')
    


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


