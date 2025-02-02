# Classification for 'diabetes.csv'
import pandas as pd
from sklearn.metrics.cluster import entropy
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier
data = pd.read_csv('diabetes.csv')
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file('report.html')


# có dữ liệu => chia x, y (features và target var)
target = 'Outcome'
x = data.drop(target, axis=1)
y = data[target]

# train = 60%, val = 20%, test = 20%
# first split train and test                                                       random seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # train: 80 %, test: 20 %
#secondly, split train into train and validation
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25) # val: 25% * 80 = 20%, train: 60%
# practically, we don't need to have validation set

# ?? what do we do when we want to evaluate what is the best model? => use test set
# next step: standardize data: fit => transform, fit_transform(do both as same time), call transform before fit() cause error
scaler = StandardScaler()
# train set often use fit_transform, test set just use transform (use fit from train set)
# never allowed to use fit() on validation and test set
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# we can't do it at first: x = scaler.fit_transform(x), cause data will be leaked (data leaked)

# next when looking at correlation matrix, what is the suitable model for this, what model should we start at?
# => we knew that |corr| > 0.7 is a strong corr, so that at the corr table, there's no strong corr
# so we should not choose linear model. Then we start at nonlinear model
# sometimes |corr| is low just because we have not removed outliers yet => test for linear model too
# Train model
# model = SVC() # support vector machine for Classification
# model = LogisticRegression()
# model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=100) # don't know the right value to make it better?
# model.fit(x_train, y_train)
#
# y_predict = model.predict(x_test)
# # for i, j in zip(y_predict, y_test):
# #     print('Prediction: {}. Actual value: {}'.format(i, j))
# print(classification_report(y_test, y_predict))

# SVC
#     accuracy                           0.73       154
#    macro avg       0.71      0.70      0.70       154
# weighted avg       0.73      0.73      0.73       154

# Logistic regression
#     accuracy                           0.75       154
#    macro avg       0.73      0.74      0.73       154
# weighted avg       0.76      0.75      0.75       154

# weighted avg cannot say which model is better, because we need to determine which is necessary for the problem
# precision or recall

# Back to the problem we don't need to separate validation set, because we don't want split data for val set,
# people always want to take advantage of data for train set and still guarantee for the val set.
# They apply 'K-fold cross validation'. eg: k = 4 / book
# And to optimize which hyperparameter is better:
# Using 'gridsearch': like a for loop to track all cases
# params = {
#     "n_estimators": [100, 200, 300],
#     "criterion": ["gini", "entropy", "log_loss"]
# }
# #                                                          not bring to params, it just a parameter not hyper
# grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=100), param_grid=params, cv=4, scoring='recall' , verbose=2) # cv: cross validation
# grid_search.fit(x_train, y_train) # after doing it, it just save the best record
# print(grid_search.best_score_) # in default, this score refer to accuracy, change it
# print(grid_search.best_params_)

# using library lazypredict
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models) # then bring best model to grid_search and continue to test


#1:31:00