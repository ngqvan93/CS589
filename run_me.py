import numpy as np
import pandas as pd
import cluster_class 
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import pairwise_distances



# Load data 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Function to measure cluster quality ---------------------
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
    


# CV for KMeans model-----------------------

# make station column into categorical variable
X_train['station'] = X_train['station'].astype('category')

# create dummy variables for all stations
train_dummy = pd.get_dummies(X_train['station'])

# column bind original variables with station indicator variables
X_train = pd.concat([X_train, train_dummy], axis=1)

# Remove station column now that we have indicators
# This avoids having any strings in the data
X_train = X_train.drop(['station'], axis=1)

# remove date variable to avoid working with dates
# already have season, holiday, work day, hour indicators 
X_train = X_train.drop(['date'], axis=1) 
#X_train['date'] = pd.to_datetime(X_train['date'])

# convert to numpy array for input to KMeans
X_train_array = np.array(X_train)

# follow the same steps for the test data set 
X_test['station'] = X_test['station'].astype('category')
test_dummy = pd.get_dummies(X_test['station'])
X_test = pd.concat([X_test, test_dummy], axis=1)
X_test = X_test.drop(['station'], axis=1)
X_test = X_test.drop(['date'], axis=1) # removes date variable
#X_test['date'] = pd.to_datetime(X_test['date'])
X_test_array = np.array(X_test)

# create dictionary of parameters for the Grid Search
# parameters are number of clusters and number of random restarts for kmeans
params = {'n_clusters':[1,40], 'random_state':[1,10]}

# initialize kmeans
km = KMeans()

# conduct grid search over ranges of hyperparameters on training data
clf = GridSearchCV(km, params)
clf.fit(X_train_array)

# grab optimal parameters; use as input for kmeans 
optimal_k = clf.get_params()
print optimal_k




# CV for classification




