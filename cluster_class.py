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

class cluster_class:
    
    def __init__(self, K, r):
        '''
        Create a cluster classifier object

        Args:
            K: Number of clusters
            r: random seed
        '''

        # Initialize a clustering model and a dictionary to contain clusters' classifier
        self.K = K
        self.kmeans = KMeans(n_clusters = self.K, random_state = r)
        self.clusters = {}
        self.clf = {}
        self.proportions = {}

    def fit_baseline(self, X):
        pass
        
    def predict_baseline(self, X, y):
        pass


    def classify_one_cluster(self, k, clf, args):
        '''
        This function fits a classifier of a specified cluster.

        Args:
            k: Cluster to classify.
            clf: A classifier (DecisionTree or SVM).
            args: Arguments of the classifier.
        '''

        X = self.clusters[k][0]
        y = self.clusters[k][1]
        clf = clf(args)
        clf.fit(X, y)
        self.clf[k] = clf


    def clustering(self, X, y):
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
                y_k = Y[ix]
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
        
        self.clustering(X, y)
        for k in range(self.K):
            self.classify_one_cluster(k, clf, args)


    def predict(self, X):
        '''
        This function makes predictions using a cluster classifier object.

        Args:
            X: A data matrix of dimension (N, D).
        '''        

        for k in xrange(self.K):
            predictions = self.clf[k].predict(X)
            prop = float(sum(predictions))/len(predictions)
            self.proportions[k] = (prop, 1-prop)

