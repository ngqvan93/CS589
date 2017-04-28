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
        self.proportions = {}

    def classify_one_cluster(self, k, clf, args):
        X = self.clusters[k][0]
        y = self.clusters[k][1]
        clf = clf(args)
        clf.fit(X, y)
        self.clf[k] = clf

    def clustering(self, X, y):
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
        self.clustering(X, y)
        for k in range(self.K):
            self.classify_one_cluster(k, clf, args)


    def predict(self, X):
        '''
        Make predictions using a cluster classifier object
        '''        
        for k in xrange(K):
            predictions = self.clf[k].predict(X)
            prop = float(sum(predictions))/len(predictions)
            self.proportions[k] = (prop, 1-prop)

    
    def score(self,X,Y):
        '''
        Compute prediction error rate for a cluster classifier object
        '''          
        Yhat = self.predict(X)
        return 1 - accuracy_score(Y,Yhat)
        