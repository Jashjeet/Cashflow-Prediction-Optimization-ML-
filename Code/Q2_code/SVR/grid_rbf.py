
import pandas as pd
import numpy as np
import os
import datetime
import time
import glob
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
import pickle
clean_df = pd.read_csv("../Data/clean_with_collector_df.csv", encoding='utf-8')

X = clean_df.loc[:, clean_df.columns != "DAYS_TO_PAY"]
y = clean_df['DAYS_TO_PAY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto', .01, .1],  \
                     'epsilon': [2, 5, 10], 'C': [1, 10, 50]}]

clf_svr_sel = GridSearchCV(svm.SVR(C=1), tuned_parameters, \
                       n_jobs = 9, scoring='neg_mean_absolute_error')
clf_svr_sel.fit(X=X_train, y=y_train)

model_filename = "svr_grid.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(clf_svr_sel, model_file)



y_pred_svr_sel=clf_svc_sel.predict(X_test)
acc_svc_sel=mean_absolute_error(y_test, y_pred_svr_sel)
print("Mean absolute error on test data is: ", mae)

