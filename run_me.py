import numpy as np
import pandas as pd
# import cluster_class 
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import pairwise_distances



# Load data 
DATA_PATH = '/Users/VanNguyen/Desktop/COMPSCI589/Final Project/Data/'

column_names = ['season','hour','holiday','workingday','weathersit',
'feeling_temp','humidity','windspeed','duration','station','type']

train = pd.read_csv(DATA_PATH + 'train.csv')
test = pd.read_csv(DATA_PATH + 'test.csv')

train.columns = column_names
test.columns = column_names

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

def make_dummy(X):
    # make station column into categorical variable
    # create dummy variables for all stations
    # column bind original variables with station indicator variables

    stations = X['station'].unique()
    dummy = pd.get_dummies(X.loc[:, 'station'])
    X = pd.concat([X, dummy], axis=1, ignore_index = True)
    X.columns = np.hstack((column_names[:-1], stations))
    X.drop('station', axis=1, inplace=True)

    return X

def KMeansCV(X, range_k, range_r):    
    # create dictionary of parameters for the Grid Search
    # parameters are number of clusters and number of random restarts for kmeans
    params = {'n_clusters': range_k, 'random_state': range_r}

    # initialize kmeans
    km = KMeans()

    # conduct grid search over ranges of hyperparameters on training data
    clf = GridSearchCV(km, params)
    clf.fit(X)

    # grab optimal parameters; use as input for kmeans 
    optimal_params = clf.best_params_
    
    return optimal_params



# CV for classification



if __name__ == '__main__':

    X_train_dummy = make_dummy(X_train)
    # X_test_dummy = make_dummy(X_test)

    range_k = range(3)
    range_r = range(5)
    best_params = KMeansCV(X = X_train_dummy, range_k = [1,2,3,4], range_r = [1,2,3,4,5])



















