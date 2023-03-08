# %%
"""
## Loan Default Classification
"""

# %%
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
import os
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import seaborn as sns


# %%
dataset = pd.read_csv('data/FinTech_Dataset.csv',index_col=0)
print(dataset.head())


# %%
dataset.shape

# %%
dataset.isna().sum()
#As there are no NaN values in this data, invalid values are not a major concern.

# %%
"""
## Data overview
"""

# %%
dataset.describe()

# %%
"""
The column labeled "Employed" is of categorical type, while the "Bank Balance" and "Annual Salary" columns are numerical. 
Our objective is to perform a binary classification task based on the target column "Defaulted."
"""

# %%
"""
## Feature engineering
"""

# %%
dataset.insert(3, 'Saving Rate', dataset['Bank Balance'] / dataset['Annual Salary'])
print(dataset.head())

# %%
"""
We generate a new feature named "Saving Rate" based on the "Bank Balance" and "Annual Salary" data. The Saving Rate feature provides insight into the spending habits of each user. Generally, a user with a higher Saving Rate is considered less likely to default. We will investigate the relationship between these variables in greater detail later on.
"""

# %%
"""
## Data distribution
"""

# %%
"""
Default distribution
"""

# %%
tbl = dataset['Defaulted?'].value_counts().reset_index()
tbl.columns = ['Status', 'Number']
tbl['Status'] = tbl['Status'].map({1 :'Defaulted', 0 :'Not defaulted'})
print(tbl)

# %%
fig = px.pie(tbl,
             values='Number', 
             names = 'Status',
             title='Default Status')
fig.show() 

# %%
"""
Loan defaults would only impact 3% of customers, creating in an imbalanced classification.
"""

# %%
"""
Employed distribution
"""

# %%
tbl = dataset['Employed'].value_counts().reset_index()
tbl.columns = ['Status', 'Number']
tbl['Status'] = tbl['Status'].map({1 :'Employed', 0 :'Unemployed'})
tbl

# %%
fig = px.pie(tbl,
             values='Number', 
             names = 'Status',
             title='Employed Status')
fig.show()

# %%
tbl = dataset.copy()
tbl['Employed'] = tbl['Employed'].replace({1 :'Employed', 0 :'Unemployed'})
tbl['Defaulted?'] = tbl['Defaulted?'].replace({1 :'Defaulted', 0 :'Not defaulted'})


# %%
fig = px.sunburst(tbl, 
                  path=['Employed','Defaulted?'],
                  title='Relationship between Employment and Loan Default')
fig.show()

# %%
"""
Contingency table
"""

# %%
tbl = pd.crosstab(dataset['Employed'],dataset['Defaulted?'])
print(tbl)

# %%
"""
Pearson’s  χ2  test for independence
"""

# %%
chi2, p, dof, ex = chi2_contingency(tbl)
print("p-value:", p)

# %%
"""
Conclusion:
As their p-value is between 0.0005 and 0.05, we draw the conclusion that they are not independent. Employed status can therefore be used to predict default.
"""

# %%
"""
Bank Balance distribution
"""

# %%
fig = px.histogram(dataset, x="Bank Balance", color='Defaulted?', 
                   marginal="box", # or violin, rug
                   hover_data=dataset.columns)
fig.show()

# %%
"""
We find that this is an asymmetric distribution, with many people having zero bank balance.

Let's further check this by calculating number of accounts with less than 10 dollars.
"""

# %%
(dataset['Bank Balance'] <= 10).sum()

# %%
"""
Conclusion:

Approximately 500 individuals have hardly saved any money in their bank accounts, which could pose a risk for loan defaults. 
Surprisingly, those who have defaulted on their loans tend to have a higher balance in their bank accounts. 
This observation may seem counterintuitive and suggests the presence of confounding factors. 
It is possible that individuals with a higher bank balance may have easier access to loans, leading to a higher number of defaults.
"""

# %%
"""
Annual Salary distribution
"""

# %%
fig = px.histogram(dataset, x="Annual Salary",
                   color="Defaulted?",
                   marginal="box", # or violin, rug
                   hover_data=dataset.columns)
fig.show()

# %%
"""
Conclusion:

1. In comparison to bank balance, there are fewer outliers when it comes to annual salary. 
2. Default cases appear to be distributed across all annual salary ranges, suggesting that annual salary may not be a reliable predictor of loan defaults.
"""

# %%
"""
Saving Rate distribution
"""

# %%
fig = px.histogram(dataset, x="Saving Rate",
                   color='Defaulted?', 
                   marginal="box", # or violin, rug
                   hover_data=dataset.columns)
fig.show()

# %%
"""
Conclusion:

The distribution of saving rate is similar to that of bank balance, but with a few extreme outliers. This suggests that people's saving habits can vary significantly. Some individuals may earn a high income but spend more than they save, while others with relatively low salaries may have a significant amount of savings.
"""

# %%
"""
## Modeling
"""

# %%
"""
Train test split
"""

# %%
RAND_SEED = 123

# %%
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,:-1], dataset.iloc[:,-1], test_size=0.3, stratify=dataset.iloc[:,-1], random_state=RAND_SEED)

# %%
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%
"""
Standardization
"""

# %%
scaler = StandardScaler().fit(X_train)

# %%
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
"""
Upsampling by SMOTE
"""

# %%
"""
During the Exploratory Data Analysis (EDA) phase, it was observed that defaulted cases constituted only 3% of the samples. 
This highly imbalanced dataset could pose a challenge for classification models that aim to minimize the cost function. 
To address this issue, the SMOTE upsampling method was introduced to rebalance the dataset.
"""

# %%
X_train.shape, y_train.shape

# %%
y_train.value_counts()

# %%
#pip install --upgrade imbalanced-learn

sm = SMOTE(random_state=RAND_SEED)
X_train, y_train = sm.fit_resample(X_train, y_train)

# %%
X_train.shape, y_train.shape

# %%
y_train.value_counts()

# %%
"""
## Classification
"""

# %%
"""
The models we will examine include </br>
Logistic Regression, </br>
Support Vector Machine, </br>
Random Forest, LightGBM, and </br>
XGboost. </br>
Our primary metric for optimization is the Recall Rate for predicting defaulted cases. </br>
This is because for a bank loan default problem, rejecting loans falsely only leads to potential interest loss, </br>
while the default of a loan leads to a significant loss of all principal.</br>
"""

# %%
"""
Logistic regression
"""

# %%
clf = LogisticRegression(solver='saga',random_state=RAND_SEED).fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

# %%
"""
Cross validation
"""

# %%
cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )

# %%
"""
First prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred))

# %%
print(classification_report(y_test,y_pred))

# %%
"""
Hyperparameter tuning
"""

# %%
from sklearn.model_selection import RandomizedSearchCV

# %%
"""
distributions = dict(C=np.linspace(2, 1000, 100),
                     penalty=['l2', 'l1'])
"""

# %%
"""
clf = RandomizedSearchCV(LogisticRegression(solver='saga',random_state=RAND_SEED), 
                         distributions,
                         scoring='recall', 
                         n_iter=100,
                         n_jobs = -1,
                         random_state=RAND_SEED)
clf_logistic = clf.fit(X_train, y_train)
clf_logistic.best_params_
"""

# %%
"""
{'penalty': 'l2', 'C': 254.02020202020202}
"""

# %%
distributions = dict(C=[254.02020202020202], penalty=['l2'])

# %%
clf = RandomizedSearchCV(LogisticRegression(solver='saga',random_state=RAND_SEED), 
                         distributions,
                         scoring='recall', 
                         n_iter=100,
                         n_jobs = -1,
                         random_state=RAND_SEED)
clf_logistic = clf.fit(X_train, y_train)
clf_logistic.best_params_

# %%
y_pred_logistic = clf_logistic.predict(X_test)

# %%
"""
Tuned prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred_logistic))

# %%
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_logistic)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# %%
print(classification_report(y_test,y_pred_logistic))

# %%
"""
Support vector machine
"""

# %%
clf = SVC(probability=True)
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

# %%
"""
Cross validation
"""

# %%
cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )

# %%
"""
First prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred))

# %%
print(classification_report(y_test,y_pred))

# %%
"""
Hyperparameter tuning
"""

# %%
"""
distributions = dict(C=np.logspace(0, 4, 50),
                     degree = np.linspace(1,10,1),
                     class_weight = [None, 'balanced'],
                    )
"""

# %%
distributions = dict(C=[494.1713361323833],
                     degree = [1.0],
                     class_weight = [None],
                    )

# %%
# For training speed the iteration is set to 1.
# Given more time we can of course train more iters.
clf = RandomizedSearchCV(SVC(probability=True, cache_size = 1024*25), 
                         distributions,
                         scoring='recall', 
                         n_iter=1, 
                         n_jobs = 1,
                         random_state=RAND_SEED) 
clf_SVC = clf.fit(X_train, y_train)
clf_SVC.best_params_

# %%
"""
{'degree': 1.0, 'class_weight': None, 'C': 494.1713361323833}
"""

# %%
y_pred_SVC = clf_SVC.predict(X_test)

# %%
"""
Tuned prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred_SVC))

# %%
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_SVC)
# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# %%
print(classification_report(y_test,y_pred_SVC))

# %%
"""
Random forest 
"""

# %%
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

# %%
"""
Cross validation
"""

# %%
cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )

# %%
"""
First prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred))

# %%
print(classification_report(y_test,y_pred))

# %%
"""
Hyperparameter tuning
"""

# %%
"""
distributions = dict(n_estimators=np.arange(10, 500, 10),
                     criterion=['gini', 'entropy'],
                     max_depth = range(20),
                     min_samples_split = range(2, 20),
                     min_samples_leaf = range(3, 50),
                     bootstrap = [True, False],
                     class_weight = ['balanced', 'balanced_subsample']
                    )
"""

# %%
"""

clf = RandomizedSearchCV(RandomForestClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=20,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_random_forest = clf.fit(X_train, y_train)
clf_random_forest.best_params_
"""

# %%
"""
{'n_estimators': 490,
 'min_samples_split': 14,
 'min_samples_leaf': 5,
 'max_depth': 8,
 'criterion': 'gini',
 'class_weight': 'balanced_subsample',
 'bootstrap': False}
"""

# %%
distributions = dict(n_estimators=[490],
                     criterion=['gini'],
                     max_depth = [8],
                     min_samples_split = [14],
                     min_samples_leaf = [5],
                     bootstrap = [False],
                     class_weight = ['balanced_subsample']
                    )

# %%
clf = RandomizedSearchCV(RandomForestClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=20,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_random_forest = clf.fit(X_train, y_train)
clf_random_forest.best_params_

# %%
y_pred_random_forest = clf_random_forest.predict(X_test)

# %%
"""
Tuned prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred_random_forest))

# %%
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_random_forest)
# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# %%
print(classification_report(y_test,y_pred_random_forest))

# %%
"""
LightGBM
"""

# %%
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

# %%
"""
Cross validation
"""

# %%
cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )

# %%
"""
First prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred))

# %%
print(classification_report(y_test,y_pred))

# %%
"""
Hyperparameter tuning
"""

# %%
"""
distributions = {
    'learning_rate': np.logspace(-5, 2, 50),
    'num_leaves': np.arange(10, 100, 10),
    'max_depth' : np.arange(3, 13, 1),
    'colsample_bytree' : np.linspace(0.1, 1, 10),
    'min_split_gain' : np.linspace(0.01, 0.1, 10),
}
"""

# %%
"""
clf = RandomizedSearchCV(lgb.LGBMClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=100,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_lgb = clf.fit(X_train, y_train)
clf_lgb.best_params_
"""

# %%
"""
{'num_leaves': 60,
 'min_split_gain': 0.030000000000000006,
 'max_depth': 8,
 'learning_rate': 0.07196856730011514,
 'colsample_bytree': 0.7000000000000001}
"""

# %%
distributions = {
    'learning_rate': [0.07196856730011514],
    'num_leaves': [60],
    'max_depth' : [8],
    'colsample_bytree' : [0.7000000000000001],
    'min_split_gain' : [0.030000000000000006],
}

# %%
clf = RandomizedSearchCV(lgb.LGBMClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=100,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_lgb = clf.fit(X_train, y_train)
clf_lgb.best_params_

# %%
y_pred_lgb = clf_lgb.predict(X_test)

# %%
"""
Tuned prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred_lgb))

# %%
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_lgb)
# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# %%
print(classification_report(y_test,y_pred_lgb))

# %%
"""
XGBoost
"""

# %%
clf = XGBClassifier()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

# %%
"""
Cross validation
"""

# %%
cross_val_score(clf, X_train, y_train, scoring='recall' ,cv=5, )

# %%
"""
First prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred))

# %%
print(classification_report(y_test,y_pred))

# %%
"""
Hyperparameter tuning
"""

# %%
"""
distributions = { 
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth':np.arange(2,10,1),
    'learning_rate':np.logspace(-4, 1, 50), 
    'subsample':np.linspace(0.1, 1, 10),
    'colsample_bytree':np.linspace(0.1, 1, 10), 
}
"""

# %%
"""
clf = RandomizedSearchCV(XGBClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=10,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_xgb = clf.fit(X_train, y_train)
clf_xgb.best_params_
"""

# %%
"""
{'subsample': 0.9,
 'n_estimators': 600,
 'max_depth': 8,
 'learning_rate': 0.008685113737513529,
 'colsample_bytree': 0.6}
"""

# %%
distributions = { 'n_estimators': [600],
                 'max_depth':[8], 
                 'learning_rate':[0.008685113737513529],
                 'subsample':[0.9],
                 'colsample_bytree':[0.6], }

# %%
clf = RandomizedSearchCV(XGBClassifier(), 
                         distributions,
                         scoring='recall', 
                         n_iter=10,
                         n_jobs = 4,
                         random_state=RAND_SEED)
clf_xgb = clf.fit(X_train, y_train)
clf_xgb.best_params_

# %%
y_pred_xgb = clf_xgb.predict(X_test)

# %%
"""
Tuned prediction result
"""

# %%
print(confusion_matrix(y_test,y_pred_xgb))

# %%
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)
# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# %%
print(classification_report(y_test,y_pred_xgb))

# %%
"""
Model assessment
"""

# %%
"""
ROC curve
"""

# %%
sns.set()

# %%
model_names = ['LogisticRegression','SVM', 'RandomForest','LightGBM','XGBoost']
models = [clf_logistic, clf_SVC, clf_random_forest, clf_lgb, clf_xgb]

plt.figure(figsize=(8, 6))

for name, model in zip(model_names, models):
    prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    model_auc = round(auc(fpr, tpr), 4)
    plt.plot(fpr,tpr,label="{}, AUC={}".format(name, model_auc))

random_classifier=np.linspace(0.0, 1.0, 100)
plt.plot(random_classifier, random_classifier, 'r--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %%
"""
Given the imbalanced nature of our dataset, our emphasis is on the precision-recall curve.
Based on the test set outcome, it can be concluded that the Logistic regression model performed well.
"""

# %%
"""
## Conclusion
"""

# %%
"""
The purpose of this notebook is to work with an imbalanced loan default dataset using multiple ML models. Our findings reveal that the Random Forest model achieved the highest Recall rate of 89% on the test set. However, the Logistic Regression model surpassed all other models with the top AUC score of 0.5238 in the precision-recall curve. With the addition of more features and feature engineering, there is potential to further enhance the results in the future.
"""

# %%
