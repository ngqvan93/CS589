import numpy as np
import pandas as pd
import cluster_class 



# Load data 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]


# CV for KMeans model





# CV for classification




