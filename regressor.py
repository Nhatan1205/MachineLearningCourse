import pandas as pd
from lazypredict.Supervised import numeric_transformer
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import  LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyRegressor
data = pd.read_csv("StudentScore.xls")
# profile = ProfileReport(data, title="Score Report", explorative=True)
# profile.to_file('score.html')
target = 'writing score'
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# remove outliers, in reality, it's rarely to remove it
# preprocessing

# print(data['gender'].unique()) check it has two or more data type, if two => boolean
# print(data['race/ethnicity'].unique()) there are 5 data type of nominal, they named group A, B, C, D, E (not a specific name) because it will not cause discrimination
# print(data['parental level of education'].unique()) ordinal type: some high school < high school < some college < associate's degree < bachelor's degree < master's degree
# print(data['lunch'].unique()) boolean

# Check which attributes are needed for writing score: cannot remove
# Suppose the data has some missing data => use SimpleImputer
# imputer = SimpleImputer(strategy='median')
# scaler = StandardScaler()
# x_train[['math score', 'reading score']] = imputer.fit_transform(x_train[['math score', 'reading score']])
# x_test[['math score', 'reading score']] = imputer.transform(x_test[['math score', 'reading score']])
#
# x_train[['math score', 'reading score']] = scaler.fit_transform(x_train[['math score', 'reading score']])
# x_test[['math score', 'reading score']] = scaler.transform(x_test[['math score', 'reading score']])

# use pipeline to short the process, just use one fit_transform
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

education_values = ["some high school", "high school", "some college",  "associate's degree", "bachelor's degree", "master's degree"]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder',  OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(sparse_output=False))  #true or false
])


# result = ord_transformer.fit_transform(x_train[['parental level of education', 'gender', 'lunch', 'test preparation course']])
# for i, j in zip(x_train[['parental level of education', 'gender', 'lunch', 'test preparation course']].values, result):
#     print("Before {}. After {}".format(i, j))
preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "math score"]),
    ("ord_feature", ord_transformer, ['parental level of education', 'gender', 'lunch', 'test preparation course']),
    ("nom_feature", nom_transformer, ['race/ethnicity'])
])

# reg = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     # ("model", LinearRegression())
#     ("model", RandomForestRegressor())
# ])
# reg.fit(x_train, y_train)
# y_predict = reg.predict(x_test)
# for i, j in zip(y_predict, y_test):
#     print('Prediction: {}. Actual value: {}'.format(i, j))

# metrics: L1, L2, r2
# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
# print("R2: {}".format(r2_score(y_test, y_predict)))

# LinearRegression
# MAE: 3.2039447691582152           the lower, the better
# MSE: 14.980822041816767           the lower, the better
# R2: 0.9378432907399291            > 0.8 is good

# RandomForestRegressor
# MAE: 3.5920740476190476
# MSE: 20.323308229705212
# R2: 0.9156768595668201

# Because there has 2 attributes having high correlation with the target so using a linear model is better than nonlinear

# For random forest,
# it has errors cause grid_search just only is fed by model, but now we are using pipeline => conflict => name before the var
# params = {
#     "preprocessor__num_feature__imputer__strategy": ['median', "mean"],
#     "model__n_estimators": [100, 200, 300],
#     "model__criterion": ["squared_error", "absolute_error", "poisson"],
#     "model__max_depth": [None, 2, 5]
# }
# #                                                          not bring to params, it just a parameter not hyper
# # grid_search = GridSearchCV(estimator=reg, param_grid=params, cv=5, scoring='r2' , verbose=2, n_jobs=-1) # cv: cross validation, n_job=-1: use full processor
# # grid_search.fit(x_train, y_train) # after doing it, it just save the best record
# # print(grid_search.best_score_) # in default, this score refer to accuracy, change it
# # print(grid_search.best_params_)
#
# # If we have many fits to test => takes a lot of time => use randomized_search
# randomize_search = RandomizedSearchCV(estimator=reg, param_distributions=params, n_iter=20, cv=5, scoring='r2' , verbose=2, n_jobs=-1) # cv: cross validation, n_job=-1: use full processor
# randomize_search.fit(x_train, y_train) # after doing it, it just save the best record
# print(randomize_search.best_score_) # in default, this score refer to accuracy, change it
# print(random
# ize_search.best_params_)

reg = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = reg.fit(x_train, x_test, y_train, y_test)
print(models) # then bring best model to grid_search and continue to test