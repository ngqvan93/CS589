import numpy as np
import pandas as pd 
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold



class cluster_class:
    
    def __init__(self, K, r):
        '''
        Create a cluster classifier object
        '''
        # Initialize a clustering model and a dictionary to contain clusters' classifier
        self.K = K
        self.kmeans = KMeans(n_clusters = self.K, random_state = r)
        self.clusters = {}
        self.clf = {}

    def classify_one_cluster(self, k, clf, *args):
        X = self.clusters[k][0]
        y = self.clusters[k][1]
        clf = clf(args)
        clf.fit(X, y)
        self.clf[k] = clf

    def clustering(self, X, Y):
        '''
        Cluster data and store to dictionary
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
                Y_k = Y[ix]
                X_k = X[ix]
                self.clusters[k] = (X_k, y_k)   
            else:
                self.labels[k] = random.choice(Y)



    def predict(self, X):
        '''
        Make predictions usins a cluster classifier object
        '''        
        pass

    
    def score(self,X,Y):
        '''
        Compute prediction error rate for a cluster classifier object
        '''          
        Yhat = self.predict(X)
        return 1 - accuracy_score(Y,Yhat)
        