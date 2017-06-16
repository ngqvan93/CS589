# Predicting Commuter Use in Bikeshare Programs
### Authors: Emma Kearney, Van Nguyen


### 1. Motivation
Bike share programs have become a popular mode of transportation in big cities. Commonly however, a rider 
may want to rent a bike only to find the nearest bicycle docking station empty, or on the contrary, try to drop off
a bike at an already full station. There are certain factors such as day of the week, holidays, or weather that can
affect the chance of this occurring.
In this project, we want to study the distribution of bikers at each station in the Washington DC metro area. This information is helpful to bike share program managers because they can anticipate the proportions of riders and maintain a sustainable system for their customers.


### 2. Data  
We obtained our data from two sources: (i) Capital Bikeshare [CBS] and (ii) UCI Machine Learning Repository [UCI] (originally constructed by Fanaee-T and Gama [FG]). The first data set from Capital Bikeshare has records of every single trip from January 2011 to December 2012. It consists of 3,288,889 observations with seven features. The second data set provides other aspects of the trips, such as dates, times, and weather conditions, broken down to every hour of the day. It has 17,379 observations with 17 features. Since both data sets have date and time as features, we merge them together by date and eliminate some features that are not relevant to the task of this project. The final full data set dimension is 3,288,889 rows by 11 columns.


### 3. Methodology
Our pipeline begins with the unsupervised learning task of clustering to identify subgroups of bike
rides with similar features, such as weather, date, and station. Then, we use supervised learning to
classify each observation within a cluster as either a registered or casual rider. To obtain cluster
labels, we calculate the proportions of registered and casual riders for each cluster. This is the final
output of our pipeline. Using the distribution of riders combined with start stations in each cluster,
we can inform Capital Bikeshare Program of the riders’ prospective usage based on weather and
temporal conditions. For clustering task, we use the K-Means clustering model. For classification, we use Decision Tree 
and Support Vector Machines techniques.


### 4. Results
In summary, the model with K-Means Clustering and Decision Tree achieve an average of 87.157% for precision and 97.035% for recall. On the other hand, K-Means when combined with SVM has an average precision of 85.840% and an average recall of 98.618%. The Decision Tree and SVM performed similarly in both performance metrics, too similarly to designate one as better. While Decision Trees are more interpretable models for bike share use, we require additional tuning of the classifier models before selecting one method over the other.
