'''
CS 589: Final Project
@authors: Emma Kearney, Van Nguyen
'''


# Load libraries ---------------------
import numpy as np
import pandas as pd 
import random
from sklearn.cluster import KMeans


# Cluster_Class object ---------------------

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
        self.pred_counts = {'R': 0, 'C': 0}
        self.raw_counts = {'R': 0, 'C': 0}
        self.true_pos = 0

    
    def fit(self, X, y, clf = None):
        '''
        This function fits clustering model and stores data in each cluster.

        Args:
            X: A data matrix of dimension (N, D).
            y: A vector label (N, 1).
            clf: A binary, True: Fit clustering/classification model, False: fit baseline model.
        '''

        # Fit a K-Means clustering model.
        self.kmeans = self.kmeans.fit(X)        
        Z = self.kmeans.labels_
        
        if clf:
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
                    r_ix = random.choice(ix)
                    self.clusters[k] = (X[r_ix], y[r_ix])


    def predict_baseline(self, X, y):
        '''
        This function make predictions from the baseline model.

        Args:
            X: A data matrix of dimension (N, D).

        Returns:
            ix_list: A 2D list. Each element is a list of indices of data observations that belong to a cluster.
            proportions: A dictionary where key is the ID of cluster and value is a tuple of proportions of riders. 
        '''

        predictions = self.kmeans.predict(X)
        ix_list = []
        for k in xrange(self.K):
            ix = np.where(predictions == k)
            ix_list.append(list(ix[0]))
            if len(ix[0]) > 0:          
                y_k = y[ix]
                prop = round(float(sum(y_k))/len(y_k), 5)
                self.proportions[k] = (prop, round(1-prop, 5))
            else:
                self.proportions[k] = (0, 0) 

        self.raw_counts['R'] = sum(y)
        self.raw_counts['C'] = len(y) - sum(y)

        return ix_list, self.proportions


    def predict(self, X, y):
        '''
        This function makes predictions using a cluster classifier object.

        Args:
            X: A data matrix of dimension (N, D).

        Returns:
            true_pos: The number of true positives, aka number of data points that are correctly labeled as "Registered".
            proportions: A dictionary where key is the ID of cluster and value is a tuple of proportions of riders. 
        '''        
        cluster_labels = self.kmeans.predict(X)

        for k in xrange(self.K):
            ix = np.where(cluster_labels == k)
            if len(ix[0]) > 0:
                X_k = X[ix]
                y_k = np.array(y[ix])
                predictions = np.array(self.clf[k].predict(X_k))
                pos_ix = np.where(predictions == 1)
                pos = predictions[pos_ix]
                y_k = y_k[pos_ix]
                errors = pos - y_k
                self.true_pos += len(np.where(errors == 0)[0])
                self.pred_counts['R'] += sum(predictions)
                self.pred_counts['C'] += len(predictions) - sum(predictions)
                
                prop = round(float(sum(predictions))/len(predictions), 5)
                self.proportions[k] = (prop, round(1-prop, 5))

        return self.true_pos, self.proportions


