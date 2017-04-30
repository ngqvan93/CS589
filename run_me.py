'''
CS 589: Final Project
@authors: Emma Kearney, Van Nguyen
'''


# Load libraries ---------------------

# Data manipulation
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt


import cluster_class 

import time

# Sklearn machine learning model
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# Model selection, evaluation metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances



# Load data ---------------------
# Define data path, column names
DATA_PATH = '/Users/VanNguyen/Desktop/COMPSCI589/Final Project/Data/'


# Load data
train = pd.read_csv(DATA_PATH + 'train.csv')
test = pd.read_csv(DATA_PATH + 'test.csv')

   

# Data Processing --------------------

def make_dummy(data):
    '''
    This function takes a full data set, converts cetegorical data to numerical data,
    and split data inputs and data outputs. 

    Args: 
        data: A data matrix of dimension (N,D+1).

    Returns:
        X: A data matrix X (N, D).
        y: A vector label (N, 1).
    '''

    # Create indicator variables for categorical features. 
    season_dummy = np.array(pd.get_dummies(data.loc[:, 'season']))
    holiday_dummy = np.array(pd.get_dummies(data.loc[:, 'holiday']))
    workingday_dummy = np.array(pd.get_dummies(data.loc[:, 'workingday']))
    weathersit_dummy = np.array(pd.get_dummies(data.loc[:, 'weathersit']))
    station_dummy = np.array(pd.get_dummies(data.loc[:, 'station']))
    type_dummy = np.array(pd.get_dummies(data.loc[:, 'type']).iloc[:, 1])

    # Delete categorical columns.
    data.drop(['season', 'holiday', 'workingday', 
        'weathersit', 'station', 'type'], axis = 1, inplace = True)

    # Column bind the dummy matrix to the data matrix.
    data = np.column_stack((np.array(data), season_dummy, station_dummy, type_dummy))
    X = data[:, :-1]
    y = data[:, -1]
    
    return X, y



# K-Means Cross Validation --------------------

def KMeans_CV(X, K_vals):   
    '''
    This function does cross validation for K-Means clustering model.

    Args:
        X: A data matrix X of dimension (N, D).
        K_vals: A range of hyperparameter values to search over.

    Returns:
        The optimal hyperparameter n_clusters of K-Means. 
    ''' 

    # Declare the range of hyperparamters to search over
    params = {'n_clusters': K_vals}

    # Initialize a K-Means model
    km = KMeans(random_state = 0)

    # Fit K-Means model and search for the best hyperparameter based on
    # K-Means default scoring method.
    km = GridSearchCV(km, params)
    km.fit(X)

    return km.best_estimator_


# Decision Tree Cross Validation --------------------

def dt_CV(k, depth_vals, X_train, y_train, create_plot = None):
    '''
    This functions runs cross validation for a Decision Tree classifier.

    Args:
        k: A positive number determining the number of fold in K-Fold cross validation.
        depth_vals: A list of values for depth of the DecisionTreeClassifier.
        X_train: A data matrix (N, D) for training and validation.
        y_train: A vector label (N, 1) for training output labels.
        create_plot: Default 'None'. If 'True', produce a plot of training and validation RMSE curves.

    Returns:
        The optimal DecisionTreeClassifier
        and an optinal plot of training and validation RMSE curves. 
    '''

    # We use negative mean squared error to account for scoring mechanism in sklearn.
    # cv = k sets the number of folds in k-fold cross validation. 

    dt = DecisionTreeClassifier(random_state = 0)
    train_scores, valid_scores = validation_curve(dt, X_train, y_train, 
                                                  param_name = "max_depth", 
                                                  param_range = depth_vals, 
                                                  cv = k, 
                                                  scoring = 'neg_mean_squared_error') 
    # Convert negative mse to useable rmse values.
    train_mse = abs(train_scores)
    train_rmse = np.sqrt(train_mse)

    valid_mse = abs(valid_scores)
    valid_rmse = np.sqrt(valid_mse)

    # Find indices of minimum CV RMSE.
    # Use the row index from above to select the corresponding alpha value.
    min_valid_rmse = np.unravel_index(valid_rmse.argmin(), valid_rmse.shape)
    optimal_depth = depth_vals[min_valid_rmse[0]]
    
    # Initialize a DecisionTreeClassifier with optimal max_depth and retrain data.
    optimal_dt = DecisionTreeClassifier(max_depth = optimal_depth) 
    optimal_dt.fit(X_train, y_train)

    # Option to produce a train/validation plot.
    if create_plot:
        plot_validation_curve(train_rmse, valid_rmse, depth_vals)

    return optimal_dt


def plot_validation_curve(train_rmse, valid_rmse, depth_vals):
    '''
    This is an utility function for dt_CV().
    It produces a train/validation curves plot during Decision Tree cross validation.
    '''
    
    # since each iteration of a new alpha value yields 5 folds,
    # take the average rmse at each alpha level
    train_rmse_mean = np.mean(train_rmse, axis=1)
    train_rmse_std = np.std(train_rmse, axis=1)
    valid_rmse_mean = np.mean(valid_rmse, axis=1)
    valid_rmse_std = np.std(valid_rmse, axis=1)
    
    # used plotting example from sklearn validation_curve documentation
    # as outline for code below
    plt.title("Train-Validation Curve with KFold-CV on Decision Tree Classifier")
    plt.xlabel("Maximum Tree Depth") 
    plt.ylabel("RMSE")
    plt.ylim(0.0, np.amax(valid_rmse)+2)
    plt.xlim(0.0, np.amax(depth_vals)+10)
    lw = 2
    plt.plot(depth_vals, train_rmse_mean, label="Training RMSE",
             color="darkorange", lw=lw)
    plt.fill_between(depth_vals, train_rmse_mean - train_rmse_std,
                     train_rmse_mean + train_rmse_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(depth_vals, valid_rmse_mean, label="Cross-validation RMSE",
             color="navy", lw=lw)
    plt.fill_between(depth_vals, valid_rmse_mean - valid_rmse_std,
                     valid_rmse_mean + valid_rmse_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    # Save plot to folder
    plt.savefig('../Figures/train_val_curve.pdf')



# SVM Cross Validation --------------------

def svm_CV(k, C_vals, kernel_vals, X_train, y_train):
    '''
    This functions runs cross validation for an SVM classifier.

    Args:
        k: A positive number determining the number of fold in K-Fold cross validation.
        C_vals: Penalty parameter C of the error term.
        kernel_vals: The kernel types to be used in the algorithm.
        X_train: A data matrix (N, D) for training and validation.
        y_train: A vector label (N, 1) for training output labels.

    Returns:
        The optimal SVC classifier.
    '''

    # Declare the range of hyperparamters to search over
    params = {'C': C_vals, 'kernel': kernel_vals}
    
    # Initialize an SVC classifier
    svc = SVC()

    # Fit SVC model and search for the best hyperparameter based on
    # SVC default scoring method.

    svc = GridSearchCV(svc, params, cv = k)
    svc.fit(X_train, y_train)
    
    return svc.best_estimator_


def main():

    # Part 1: Make data.
    X_train, y_train = make_dummy(train)
    X_test, y_test = make_dummy(test)

    
    # Part 2: K-Means CV.   
    K_vals = range(1, 40)
    best_km = KMeans_CV(X = X_train, K_vals = K_vals)
    # Note down best K here.
    print best_km



    # Part 3: Fit pipeline.
    
    best_K = 10                                    # CHANGE THIS 

    # Part 3.1: Fit baseline model.
    clf = cluster_class.Cluster_Class(K = best_K, r = 0)
    clf.fit_baseline(X_train, y_train)
    clf.predict_baseline(X_test, y_test)
    print clf.proportions

    # Part 3.2: Fit cluster-classification model. Tune each classifier.
    
    # Part 3.2.1: Tune DecisionTreeClassifier.
    depth_vals = range(1, 10)          # CHANGE THIS
    
    clf = cluster_class.Cluster_Class(K = best_K, r = 0)
    clf.fit_baseline(X_train, y_train)
    
    for i in xrange(clf.K):
        best_dt = dt_CV(k = 3, #number of folds
            depth_vals = depth_vals, 
            X_train = clf.clusters[i][0],
            y_train = clf.clusters[i][1], 
            create_plot = None)
        clf.clf[i] = best_dt

    predictions = clf.predict(X_test)
    print predictions

    # # Part 3.2.2: Tune SVM classifier.
    # C_vals = [0.0001, 0.001, 0.1, 1, 10, 20, 30, 50, 100, 500, 1000]          # CHANGE THIS
    # kernel_vals = ['rbf', 'linear']
    
    # clf = cluster_class.Cluster_Class(K = best_K, r = 0)
    # clf.fit_baseline(X_train, y_train)
    
    # best_C = []
    # best_kernels = []
    # for i in xrange(clf.K):
    #     best_svm = svm_CV(k = 3, # number of folds
    #         C_vals = C_vals, 
    #         kernel_vals = kernel_vals, 
    #         X_train = clf.clusters[i][0], 
    #         y_train = clf.clusters[i][1])
    #     clf.clf[i] = svm

    # predictions = clf.predict(X_test)
    # print predictions


if __name__ == '__main__':

    main()
        






