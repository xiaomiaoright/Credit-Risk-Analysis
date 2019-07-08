# NaiveBayes
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr

%matplotlib inline

"""
Data Prepared:
Cleaning missing values.
Aggregating categories of certain categorical variables.
Encoding categorical variables as binary dummy variables.
Standardizing numeric variables.
"""

## -->> load dataset
Features = np.array(pd.read_csv('Credit_Features.csv'))
Labels = np.array(pd.read_csv('Credit_Labels.csv'))
Labels = Labels.reshape(Labels.shape[0],)
print(Features.shape)
print(Labels.shape)

# The Features array has both numeric features and binary features 
# (dummy variables for the categorical features). 
# Therefore, a Gaussian model must be used. 
# numeric features are mixed with features exhibiting Bernoulli distributions, the binary feature

nr.seed(321)
cv_folds = ms.KFold(n_splits=10, shuffle = True)
    
nr.seed(498)
NB_credit = GaussianNB()
cv_estimate = ms.cross_val_score(NB_credit, Features, Labels, 
                                 cv = cv_folds) # Use the outside folds

print('Mean performance metric = %4.3f' % np.mean(cv_estimate))
print('SDT of the metric       = %4.3f' % np.std(cv_estimate))
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate):
    print('Fold %2d    %4.3f' % (i+1, x))



nr.seed(498)
NB_credit = GaussianNB()
cv_estimate = ms.cross_val_score(NB_credit, Features, Labels, 
                                 cv = 10) # Use the outside folds

print('Mean performance metric = %4.3f' % np.mean(cv_estimate))
print('SDT of the metric       = %4.3f' % np.std(cv_estimate))
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate):
    print('Fold %2d    %4.3f' % (i+1, x))
#  this model is likely to generalize well.


## -->>build and test a model using a single split of the dataset. 
## Randomly sample cases to create independent training and test data
nr.seed(1115)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 300)
X_train = Features[indx[0],:]
y_train = np.ravel(Labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(Labels[indx[1]])

# Define and fit the model
NB_credit_mod = GaussianNB() 
NB_credit_mod.fit(X_train, y_train)

# Test the model
def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])

def print_metrics(labels, probs, threshold):
    scores = score_model(probs, threshold)
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))
    print('AUC             %0.2f' % sklm.roc_auc_score(labels, probs[:,1]))
    print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1]))/2.0))
    print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1]))/2.0))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
    
probabilities = NB_credit_mod.predict_proba(X_test)
print_metrics(y_test, probabilities, 0.5)   
## Ahhh...  these performance metrics are poor.

## try  Bernoulli naive Bayes model 
#  naive Bayes models tend to be less sensitive to the quantity of training data, this approach may be reasonable
# To apply this model, the numeric features must be dropped from the array. 
#  remove the numeric features 
Features = Features[:,4:]
Features[:3,:]


## -->> Nested Cross Validation
nr.seed(123)
inside = ms.KFold(n_splits=10, shuffle = True)
nr.seed(321)
outside = ms.KFold(n_splits=10, shuffle = True)


nr.seed(3456)
## Define the dictionary for the grid search and the model object to search on
param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
## Define the NB regression model
NB_clf = BernoulliNB() 

## Perform the grid search over the parameters
clf = ms.GridSearchCV(estimator = NB_clf, param_grid = param_grid, 
                      cv = inside, # Use the inside folds
                      scoring = 'roc_auc',
                      return_train_score = True)
clf.fit(Features, Labels)
print(clf.best_estimator_.alpha)

#NB_credit = BernoulliNB(alpha = clf.best_estimator_.alpha)
nr.seed(498)
cv_estimate = ms.cross_val_score(clf, Features, Labels, 
                                 cv = outside) # Use the outside folds

print('Mean performance metric = %4.3f' % np.mean(cv_estimate))
print('SDT of the metric       = %4.3f' % np.std(cv_estimate))
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate):
    print('Fold %2d    %4.3f' % (i+1, x))

## -->> Build and test BernoulliNB model
## Randomly sample cases to create independent training and test data
nr.seed(1115)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 300)
X_train = Features[indx[0],:]
y_train = np.ravel(Labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(Labels[indx[1]])

# Define and fit the model
NB_credit_mod = BernoulliNB(alpha = clf.best_estimator_.alpha) 
NB_credit_mod.fit(X_train, y_train)
probabilities = NB_credit_mod.predict_proba(X_test)
print_metrics(y_test, probabilities, 0.5)    
# The results for this Bernoulli naive Bayes model are much better than for the Gaussian model. 

# The current model uses the empirical distribution of the label values for the prior value of  𝑝  of the Bernoulli distribution. 
# This probability is invariably skewed toward the majority case. 
# Since the bank cares more about the minority case, setting this distribution to a fixed prior value can help overcome the class imbalance. 
# The code in the cell below redefines the model object with prior probability of 0.6 for the minority case.
NB_credit_mod = BernoulliNB(alpha = clf.best_estimator_.alpha,
                            class_prior = [0.4,0.6]) 
NB_credit_mod.fit(X_train, y_train)
probabilities = NB_credit_mod.predict_proba(X_test)
print_metrics(y_test, probabilities, 0.5)    
# Yeah! The majority of bad credit cases are now correctly identified.