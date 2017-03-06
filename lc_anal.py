# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:52:13 2016

@author: lenovo1
"""

import pandas as pd
import numpy as np
import datetime 
from pandas.io.data import DataReader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import preprocessing

from datetime import datetime

#Load data

loans1 = pd.read_csv('C:/doc/lcdata/LoanStats3a1.csv', delimiter=',')
loans1f = loans1[loans1.loan_status == 'Charged Off']
loans1p = loans1[loans1.loan_status == 'Fully Paid']
loans2 = pd.read_csv('C:/doc/lcdata/LoanStats3b1.csv', delimiter=',')
loans2f = loans2[loans2.loan_status == 'Charged Off']
loans2p = loans2[loans2.loan_status == 'Fully Paid']
#loans3 = pd.read_csv('C:/doc/lcdata/LoanStats3c1.csv', delimiter=',')
#loans4 = pd.read_csv('C:/doc/lcdata/LoanStats3d1.csv', delimiter=',')
#loans5 = pd.read_csv('C:/doc/lcdata/LoanStats_2016Q1a.csv', delimiter=',', low_memory=False)
#loans6 = pd.read_csv('C:/doc/lcdata/LoanStats_2016Q2a.csv', delimiter=',', low_memory=False)

loans = pd.concat([loans1f, loans1p, loans2f, loans2p])

#Prepare dataset to train model

from collections import Counter

word_list = []

for i in range(0,  len(loans)):
    sent = loans['desc'].iloc[i]
    if type(sent) is str:    
        words = sent.split()
    for word in words:    
        word_list.append(word)

cnt_wordlist = Counter(word_list)
    
#watch_words = ['Borrower', 'credit', 'pay', 'loan', 'will', 'debt', 'interest', 'card', 'cards', 'consolidate', 'payment', 'high', 'paying', 'help']
watch_words = ['help', 'need', 'bills']

def check_word(word, x):
    if type(x) is str:
       if word in x:
          return 1
    return 0

for word in watch_words:
    loans[word] = loans['desc'].apply(lambda x: check_word(word, x))
    
#ya = []

#for i in range(0, len(loans)):
#    if loans.iloc[i, 16] == 'Charged Off':
#        ya.append(1)
#    else:
#        ya.append(0)


#def 

#def num_status(row):
#    if row['loan_status'] == 'Charged Off':
#        return 1
#    else:
#        return 0
        
#loans['status'] = loans.apply(num_status, axis=1)
        
#Train model        
        
precision = dict()
recall = dict()
average_precision = dict()

ya = (loans['loan_status'] == 'Charged Off') * 1

x_dti = loans['dti']
x_inc = loans['annual_inc']

x_words = loans['help']

for word in watch_words[1:]:
    x_words = pd.concat([x_words, loans[word]], axis=1)

x_rate = (loans['int_rate'])
x_rate = (x_rate.apply(lambda x: x.strip(' %')))
x_rate = (x_rate.apply(float))

x_subgrade = loans['sub_grade']
x_grade = loans['grade']
x_subgrades = pd.get_dummies(data=x_subgrade, prefix='Grade_', dummy_na=True)
x_grades = pd.get_dummies(data=x_grade, prefix='Grade_', dummy_na=True)

x_purpose = loans['purpose']
x_emp_len = loans['emp_length']
x_state = loans['addr_state']
x_amount= loans['loan_amnt']
x_verified =(loans['verification_status'] == 'Verified')
x_pub_rec = loans['pub_rec']
x_open_acc = loans['open_acc']
x_revol_util = loans['revol_util'].apply(lambda x: x.strip('%'))
x_total_acc = loans['total_acc']

loans['issue_date'] = pd.to_datetime(loans['issue_d'])
loans['earliest_cr_line_date'] = pd.to_datetime(loans['earliest_cr_line'])


x_days_credit = loans['issue_date'] - loans['earliest_cr_line_date']
x_mths_credit = x_days_credit.astype('timedelta64[M]').astype(int)


x_inquiries = loans['inq_last_6mths']
x_dlq_2years = loans['delinq_2yrs']

x_purpose_v = pd.get_dummies(data=x_purpose, prefix='Purpose_', dummy_na = True)
x_emp_len_v = pd.get_dummies(data=x_emp_len, prefix='Emp_', dummy_na=False)
x_state_v = pd.get_dummies(data=x_state, prefix='State_', dummy_na=True)

xa = pd.concat([x_amount, x_mths_credit, x_verified, x_pub_rec, x_open_acc, x_revol_util, x_total_acc, x_inquiries, x_dlq_2years, x_purpose_v, x_dti, x_inc, x_rate], axis=1)
#xa = x_purpose_v


Xg = pd.np.array(x_grades)
Xsg = pd.np.array(x_subgrades)
Xintrate = pd.np.array(x_rate)

Xo = pd.np.array(xa)

Y = pd.np.array(ya)

l = Xintrate.shape[0]

X_strain, X_stest, Y_strain, Y_stest = train_test_split(Xsg, Y, test_size=0.33, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(Xg, Y, test_size=0.33, random_state=42)
X_itrain, X_itest, Y_itrain, Y_itest = train_test_split(Xintrate, Y, test_size=0.33, random_state=42)

X_otrain, X_otest, Y_otrain, Y_otest = train_test_split(Xo, Y, test_size=0.33, random_state=42)

logistic_subgrade = LogisticRegression()
logistic_grade = LogisticRegression()
logistic_intrate = LogisticRegression()

logistic_other = LogisticRegression()

linear_intrate = LinearRegression()

treemodel_intrate = tree.DecisionTreeClassifier()

forestmodel = RandomForestClassifier(n_estimators = 33, max_depth = None)

model_subgrade = logistic_subgrade.fit(X = X_strain, y = Y_strain, sample_weight = None)
model_grade = logistic_grade.fit(X = X_train, y = Y_train, sample_weight = None)
model_intrate = logistic_intrate.fit(X = X_itrain.reshape(len(X_itrain), 1), y = Y_itrain, sample_weight = None)
model_other = logistic_other.fit(X = X_otrain, y = Y_otrain, sample_weight = None)

lin_intrate = linear_intrate.fit(X = X_itrain.reshape(len(X_itrain), 1), y = Y_itrain, sample_weight = None)

tree_intrate = treemodel_intrate.fit(X = X_itrain.reshape(len(X_itrain), 1), y = Y_itrain, sample_weight = None)

forest_other = forestmodel.fit(X = X_otrain, y = Y_otrain, sample_weight = None)

y_pred_subgrade = model_subgrade.predict_proba(X = X_stest)
y_pred_grade = model_grade.predict_proba(X = X_test)
y_pred_intrate = model_intrate.predict_proba(X = X_itest.reshape(len(X_itest), 1))
y_pred_other = model_other.predict_proba(X = X_otest)

y_pred_linintrate = lin_intrate.predict(X = X_itest.reshape(len(X_itest), 1))

y_pred_trintrate = tree_intrate.predict_proba(X = X_itest.reshape(len(X_itest), 1))

y_pred_forest_other = forest_other.predict_proba(X = X_otest)

#plt.hist(y_pred_subgrade[:, 1])
#plt.hist(y_pred_grade[:, 1])


#precision, recall, thresholds = precision_recall_curve(Y_test, ypred[:, 1])   


#plt.clf()
#plt.plot(recall, precision, label='Precision-Recall curve')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('Precision-Recall'.format(precision[0]))
#plt.legend(loc="lower left")
#plt.show()


#fpr, tpr, thresholds = roc_curve(Y_stest, y_pred_subgrade[:, 1], pos_label = 1)

#auc_subgrade = auc(fpr, tpr)


#plt.plot(fpr, tpr, label='ROC curve subgrades (area=%0.3f)' % auc_subgrade)

#fpr, tpr, thresholds = roc_curve(Y_test, y_pred_grade[:, 1], pos_label = 1)

#auc_grade = auc(fpr, tpr)

#plt.plot(fpr, tpr, label='ROC curve grades (area=%0.3f)' % auc_grade)

plt.clf()

fpr, tpr, thresholds = roc_curve(Y_itest, y_pred_trintrate[:, 1], pos_label = 1)

auc_intrate = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve borrower rates only (area=%0.3f)' % auc_intrate)

fpr, tpr, thresholds = roc_curve(Y_otest, y_pred_forest_other[:, 1], pos_label = 1)

auc_other = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve random forest model (area=%0.3f)' % auc_other)



plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('ROC'.format(fpr[0]))
plt.legend(loc="upper left")
plt.show()



#Calculate return of LC loans

from time import strptime
from datetime import datetime
import math

strptime('Feb','%b').tm_mon

loans['net_cash_flow'] = loans['total_pymnt'] - loans['funded_amnt']

loans['last_pymnt_mth'] = loans['last_pymnt_d'].str[0:3]
loans['last_pymnt_yr'] = loans['last_pymnt_d'].str[4:8]

loans['issue_mth'] = loans['issue_d'].str[0:3]
loans['issue_yr'] = loans['issue_d'].str[4:8]

loans3 = loans.ix[np.random.choice(loans.index.values, 50000)]

loansx = loans3[loans3['last_pymnt_d'].apply(lambda x: type(x) == str)]
loansy = loansx[loansx['issue_d'].apply(lambda x: type(x) == str)]
loansz = loansy[loansy['total_pymnt'].apply(lambda x: x > 0.0)]
loansy = loansz[loansz['funded_amnt'].apply(lambda x: x > 0.0)]

loansy['last_pymnt_dt'] = loansy['last_pymnt_d'].apply(lambda x: datetime.strptime(x, "%b-%Y"))
loansy['issue_dt'] = loansy['issue_d'].apply(lambda x: datetime.strptime(x, "%b-%Y"))

loansy['duration'] = (loansy['last_pymnt_dt'] - loansy['issue_dt'])
loansy['duration_days'] = loansy['duration'].apply(lambda x: x.days)

loansy['ratio'] = loansy['total_pymnt_inv'] / loansy['funded_amnt']
loansy['inv_days'] = 1.0 / loansy['duration_days']

loansy['return_rate'] = (loansy['ratio'] ** (loansy['inv_days'] * 365) - 1)

mean_return = (loansy.groupby(['sub_grade'])['return_rate'].mean()).rename("Mean")

def num_status(row):
    if row['loan_status'] == 'Charged Off':
        return 1
    else:
        return 0
       
loansy['status'] = loansy.apply(num_status, axis=1)

mean_unit_choffrate = (loansy.groupby(['sub_grade'])['status'].mean()).rename("Mean")

loansy['int_rate_num'] = loansy['int_rate'].apply(lambda x: float(x.strip('%')))

mean_intrate = (loansy.groupby(['sub_grade'])['int_rate_num'].mean()).rename("Mean")