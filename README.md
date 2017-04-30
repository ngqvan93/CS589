# Predicting Commuter Use in Bikeshare Programs
### Authors: Emma Kearney, Van Nguyen


### 1. Problem  
Bike share programs have become a popular mode of transportation in big cities. Commonly however, a rider 
may want to rent a bike only to find the nearest bicycle docking station empty, or on the contrary, try to drop off
a bike at an already full station. There are certain factors such as day of the week, holidays, or weather that can
affect the chance of this occurring. Not surprisingly, data collection on these programs is easy and inexpensive,
thus various visualization and analysis techniques have been attempted to resolve this issue.
In this project, we will perform a supervised learning task, with the goal of building a regression model
that predicts the proportions of registered and casual bikers at all bike stations in the Washington DC metro
area. If bike share programs can anticipate the number of registered riders (commuters), they can stock stations
accordingly and maintain a sustainable system for both kinds of bikers.


### 2. Data  
Our data are taken from two sources. The UCI Machine Learning repository provides daily logs with weather
conditions and counts of each type of biker. The Capital Bikeshare Program website provides more detailed
records of every trip with durations, start and end stations, and type of biker. We will combine these two sources
and summarize the data into one final data set, where the features will be date, start station, end station, various
weather conditions, and counts of casual and registered bikers. The links to the two sources of data are in the
references section.


### 3. Methodology
This is a supervised learning problem with a regression task. Using inputs such as start and end stations,
weather conditions, and indicator of working day, we will predict the proportion of registered bikers as outputs.
The process will be as follows:
* Merge the two data sets to derive a full final data set with only the features we want.
* Split the final data into a learning set and a test set.
* Implement a pipeline that takes in a regression method with default parameters, performs feature selection,
and optimizes hyperparameters. The pipeline will output an optimized model.
* Retrain the optimized model on our full learning set, then make predictions on the held out test set.
We will complete data wrangling on local hardware using R and the code libraries plyr and dplyr. We will use
Python on local hardware to build and implement the pipeline, relying on numpy and scikit-learn code libraries.


### 4. Experiments
We will implement K-fold cross-validation. This requires randomly shuffling all observations (with corresponding
labels), then partitioning the data set into learning and test sets using an 80/20 split. The pipeline
will optimize hyperparameters by performing k-fold cross-validation on the learning set, then select the model
that yields the lowest RMSE. We will also consider several model selection criteria such as AIC, BIC, Cp, etc.
Once we use our optimized model to predict on the test set, we will assess the model’s performance based on
RMSE. If we obtain an unsatisfactory RMSE, we will adjust our pipeline by changing the structure, switching
the base regression model, or changing the number of k-folds.


### 5. Related Work and Novelty
Hadi Fanaee-T and Joao Gama augmented the original Capital Bikeshare program data with weather and
holiday information. Then, using the logs of the bike sharing system, they built an event labeling system that
relies on an ensemble of detectors and background knowledge. Another study has been done by Romain Giot
and Raphael Cherrier. They analyzed various regression models that predict the use of a bike sharing system
for the next twenty-four hours at a frequency of one hour. Lastly, Tanner Gilligan and Jean Kono constructed
a custom linear regression model to impute missing data from time series bike-rental records. Other than these
academic works, there are many data challenges hosted in different cities throughout the country that utilized
the data from each city’s bike sharing system.
We have researched most of these works and believe that the task we are trying to resolve has not been
done before. In the spirit of trying to improve the efficiency of the bike share program, we want to predict the
distribution of bike usage, split between two types of bikers at every station in Washington DC.


### 6. References
[FG] H. Fanaee-T, J. Gama. Event labeling combining ensemble detectors and background knowledge. Progress
in Artificial Intelligence. Springer Berlin Heidelberg. pp. 1-15. 2013.  
[GC] R. Giot, R. Cherrier. Predicting Bikeshare System Usage Up to One Day Ahead. IEEE Symposium Series
in Computational Intelligence 2014 (SSCI 2014). Workshop on Computational Intelligence in Vehicles and
Transportation Systems (CIVTS 2014). pp. 1-8. Dec 2014.  
[GK] T. Gilligan, J. Kono. Prediction of Bike Rentals. Web. 2014.  
Data resources:
* UCI Repository: http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
* Capital Bikeshare Program: https://s3.amazonaws.com/capitalbikeshare-data/index.html
