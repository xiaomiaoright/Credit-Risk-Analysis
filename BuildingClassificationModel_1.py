# The credit risk anlaysis problem is a two-class classification problem
# Supervise machine learning
# Modeling with basic logistic regression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

%matplotlib inline

# Load and prepare dataset
credit = pd.read_csv('German_Credit_Preped.csv')
print(credit.shape)
credit.head()

# Examine class imbalance problem
credit_counts = credit['bad_credit'].value_counts()
print(credit_counts)


##-->> Data Preparation for scikit-learn model: arrays
# create label array
labels = np.array(credit['bad_credit'])

# create features array
# encode categorical features, transform to dummy variables, append each categorical variables
def encode_string(cat_features):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

categorical_columns = ['credit_history', 'purpose', 'gender_status', 
                       'time_in_residence', 'property']

Features = encode_string(credit['checking_account_status'])
for col in categorical_columns:
    temp = encode_string(credit[col])
    Features = np.concatenate([Features, temp], axis = 1)

print(Features.shape)
print(Features[:2, :])    

# numeric features must be concatenated to the feature numpy array
Features = np.concatenate([Features, np.array(credit[['loan_duration_mo', 'loan_amount', 
                            'payment_pcnt_income', 'age_yrs']])], axis = 1)
print(Features.shape)
print(Features[:2, :]) 

##-->> Split train-test set
## Randomly sample cases to create independent training and test data
nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 300)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

# Scale numerical features
scaler = preprocessing.StandardScaler().fit(X_train[:,34:])
X_train[:,34:] = scaler.transform(X_train[:,34:])
X_test[:,34:] = scaler.transform(X_test[:,34:])
X_train[:2,]

## -->> Create logistic regression model
logistic_mod = linear_model.LogisticRegression() 
logistic_mod.fit(X_train, y_train)

# examine the model coefficients 
print(logistic_mod.intercept_)
print(logistic_mod.coef_)

# compute and display a sample of these class probabilities for the test feature set
probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])


## -->> Score and evaluate the classification model
#  set the threshold between the two likelihoods at  0.5
def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])

# Score metrics: Confusion matrix, accuracy, Precision, Recall, F1
def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

print_metrics(y_test, scores)   
# Result not good
"""
1. The confusion matrix shows the following characteristics; 
a) most of the positive cases are correctly classified, 182 vs. 30, however, 
b) may negative cases are are scored incorrectly, with only 49 correct, vs. 39 incorrect.
2. accuracy is 0.77--> Misleading!!  the negative cases are poorly classified
3. The class imbalance is confirmed. Of the 300 test cases 212 are positive and 88 are negative.
4. The precision, recall and F1 all show that positive cases are classified reasonably well, but the negative cases are not.
But negative cases are of greatest importance to the bank!!
"""

# Examine AUC and ROC
def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    
    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
plot_auc(y_test, probabilities)    

## -->> COmpute a weighted model -- deal with imbalance problem
# weight the classes when computing the logistic regression model
logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.45, 1:0.55}) 
logistic_mod.fit(X_train, y_train)

probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

scores = score_model(probabilities, 0.5)
print_metrics(y_test, scores)  
plot_auc(y_test, probabilities)  
# The accuracy is slightly changed with respect to the unweighted model.

## -->> How to find the better threshold?
def test_threshold(probs, labels, threshold):
    scores = score_model(probs, threshold)
    print('')
    print('For threshold = ' + str(threshold))
    print_metrics(labels, scores)

thresholds = [0.45, 0.40, 0.35, 0.3, 0.25]
for t in thresholds:
    test_threshold(probabilities, y_test, t)
# 0.5 is better 

