import numpy as np
import pandas as pd 
import random
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

class cluster_class:
    
    def __init__(self, K, r):
        '''
        Create a cluster classifier object
        '''
        # Initialize a clustering model and a dictionary to contain class labels
        self.K = K
        self.kmeans = KMeans(n_clusters = self.K, random_state = r)
        self.labels = {}


    def fit_cluster(self, X, Y, clf, *args):
        '''
        Learn a cluster classifier object
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

                # Find the labels of data in cluster kth.
                Y_k = Y[ix]
                # Subset X in cluster kth.
                X_k = X[ix]
                # Append the labels to X.
                X_k['label'] = Y_k
                # Group by station.
                groups = X[ix].groupby('station')

                # Iterate through each group of station. 
                # Fit a classification model.
                for g in groups.groups:
                    X_kg = groups.get_group(g)
                    X_kg = X_kg[X_kg.columns.difference('station')] # drop station column
                    Y_kg = X_kg['label']
                    clf = clf(*args[k]).fit(X_kg, Y_kg)
                    labels[k][g] = clf
            else:
                self.labels[k] = random.choice(Y)


    def cluster_quality(X,Z,K):
    '''
    Compute a cluster quality score given a data matrix X (N,D), a vector of 
    cluster indicators Z (N,), and the number of clusters K.
    '''
    
    cluster_ss = 0
    
    for k in xrange(K):
        ix = np.where(Z==k)
        # Check if there is any case assigned to cluster kth.
        if len(ix[0]) > 0:
            X_k = X[ix]
            pw_dist = pairwise_distances(X_k)
            sum_pw_dist = np.sum(np.tril(pw_dist))
        # If not, set the sum of pairwise distances to be 0.
        else:
            sum_pw_dist = 0
        cluster_ss += (1/float(len(ix)))*sum_pw_dist
    
    return cluster_ss
        
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
        