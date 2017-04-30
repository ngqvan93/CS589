'''
CS 589: Final Project
@authors: Emma Kearney, Van Nguyen
'''


# Load libraries ---------------------
import numpy as np
import pandas as pd 
import random
from sklearn.cluster import KMeans


# Cluster_class object ---------------------

class Cluster_Class:
    
    def __init__(self, K, r):
        '''
        This function creates a cluster classifier object.

        Args:
            K: Number of clusters.
            r: random seed.
        '''

        # Initialize a clustering model and a dictionary to contain clusters' classifier
        self.K = K
        self.kmeans = KMeans(n_clusters = self.K, random_state = r)
        self.clusters = {}
        self.clf = {}
        self.proportions = {}


    def classify_one_cluster(self, k, clf, *args):
        '''
        This function fits a classifier of a specified cluster.

        Args:
            k: Cluster to classify.
            clf: A classifier (DecisionTree or SVM).
            args: Arguments of the classifier.
        '''

        X = self.clusters[k][0]
        y = self.clusters[k][1]
        if clf == DecisionTreeClassifier:
            clf = clf(max_depth = args[0])
        else:
            clf = clf(C = args[0], kernel = args[1])
        clf.fit(X, y)
        self.clf[k] = clf


    def fit_baseline(self, X, y):
        '''
        This function fits clustering model and stores data in each cluster.

        Args:
            X: A data matrix of dimension (N, D).
            y: A vector label (N, 1).
        '''

        # Fit a K-Means clustering model.
        self.kmeans = self.kmeans.fit(X)        
        Z = self.kmeans.labels_
        
        # Iterate through each cluster.         
        for k in range(self.K):
            # Find index of cluster k.
            ix = np.where(Z==k)

            # Check if there is any case assigned to cluster kth.
            if len(ix[0]) > 0:
                y_k = y[ix]
                X_k = X[ix]
                self.clusters[k] = (X_k, y_k)   
            else:
                label = random.choice(y)
                if label == 1:
                    self.proportions[k] = (1, 0)
                else:
                    self.proportions[k] = (0, 1)


    def fit(self, X, y, clf, args):
        '''
        This function fits data based on the pipeline.

        Args:
            X: A data matrix of dimension (N, D).
            y: A vector label (N, 1).
            clf: A classifier (DecisionTree or SVM).
            args: Arguments of the classifier.
        '''
        
        self.fit_baseline(X, y)
        for k in range(self.K):
            self.classify_one_cluster(k, clf, args[k])


    def predict_baseline(self, X):
        '''
        This function make predictions from the baseline model.

        Args:
            X: A data matrix of dimension (N, D).

        Returns:
            A dictionary where key is the ID of cluster and value is a tuple of proportions of riders. 
        '''

        self.kmeans.predict(X)
        for k in xrange(self.K):
            if len(self.clusters[k][1]) > 0:
                prop = float(sum(self.clusters[k][1]))/len(self.clusters[k][1])
                self.proportions[k] = (prop, 1-prop) 

        return self.proportions


    def predict(self, X):
        '''
        This function makes predictions using a cluster classifier object.

        Args:
            X: A data matrix of dimension (N, D).

        Returns:
            A dictionary where key is the ID of cluster and value is a tuple of proportions of riders. 
        '''        

        for k in xrange(self.K):
            if len(self.clusters[k][1]) > 0:
                predictions = self.clf[k].predict(X)
                prop = float(sum(predictions))/len(predictions)
                self.proportions[k] = (prop, 1-prop)

        return self.proportions