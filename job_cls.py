import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN
from sklearn.model_selection import GridSearchCV

def filter_location(location):
    result = re.findall('\,\s[A-Z]{2}$', location)
    if len(result):
        return result[0][2:]
    else:
        return location

data = pd.read_excel('final_project.ods', engine="odf", dtype=str)
# print(data["career_level"].value_counts())  so imbalanced data
# print(data.isna().sum()) # test missing value or not
# title           0
# location        0
# description     1             if just one, simply remove it
# function        0
# industry        0
# career_level    0
# dtype: int64
data = data.dropna(axis=0)
# print(data.isna().sum()) # test missing value or not
data['location'] = data['location'].apply(filter_location)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
# ros = RandomOverSampler(random_state=0)
# x, y = ros.fit_resample(x, y) # has problem when we put it in there, it affects on test set where some ft identical in both train and test set => data leaked
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)
# print(y_train.value_counts())  #the percentage of test and train size is true but the percentage of each class is not followed by this %
# print(y_test.value_counts())   #=> we have stratified
print(y_train.value_counts())
print("----------------------------------")
# sampling_strategy helps to control the number of samples
# ros = RandomOverSampler(random_state=0, sampling_strategy={
#     "bereichsleiter": 1000,
#     "director_business_unit_leader": 500,
#     "specialist": 500,
#     "managing_director_small_medium_company": 500
# })
ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy={
    "bereichsleiter": 1000,
    "director_business_unit_leader": 500,
    "specialist": 500,
    "managing_director_small_medium_company": 500
})
x_train, y_train = ros.fit_resample(x_train, y_train) # just do for train set
print(y_train.value_counts())


# preprocessing
# title: Count vectorizer + tf/idf => TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words='english')
# result = vectorizer.fit_transform(x_train['title'])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result.shape)
# AttributeError: 'int' object has no attribute 'lower' => map everything to string

# location:
# encoder = OneHotEncoder() there are too many different columns
# use the code of each place => regular expression
# encoder = OneHotEncoder()
# result = encoder.fit_transform(x_train[['location']])
# print(result.shape)

# description: tf/idf
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
# result = vectorizer.fit_transform(x_train['description'])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result.shape)
# uni: (6458, 66674)
# uni + bi: (6458, 846809)
# why from uni to uni + bi it has alot token added, should've approximate? because this uni just has unique each word

# function column:
# print(len(data['function'].unique())) # 19 => ok to use one hot vector, normally < 50

# industry function
# print(len(data['industry'].unique())) # 352 too much to use one hot => use tf/idf

preprocessor = ColumnTransformer(transformers=[
    # tfidf cant use []
    ("title_ft", TfidfVectorizer(stop_words='english',ngram_range=(1,1)), 'title'),
    ("location_ft", OneHotEncoder(handle_unknown='ignore'), ["location"]),
    ("des_ft", TfidfVectorizer(stop_words='english',ngram_range=(1,2), min_df=0.01, max_df=0.95), 'description'),
    ("function_ft", OneHotEncoder(handle_unknown='ignore'), ["function"]),
    ("industry_ft", TfidfVectorizer(stop_words='english',ngram_range=(1,1)), 'industry'),
])
cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    # ("feature_selector", SelectKBest(chi2, k=200)), # when we use SelectKBest, we need to know how many ft we have and how many ft we want to remain (10% total)
    ("feature_selector", SelectPercentile(chi2, percentile=5)), # 5%
    ("model", RandomForestClassifier())
])

params = {
    # "model__n_estimators": [100, 200, 300],
    "model__criterion": ["gini", "entropy", "log_loss"],
    "feature_selector__percentile": [1,5,10]
}
grid_search = GridSearchCV(estimator=cls, param_grid=params, cv=4, scoring='recall_weighted' , verbose=2) # There's an error "ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted']."
# Recall, f1 or precision just use for binary class targeted. In this case, the target has multi class => Use f1_macro,...
grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
# result = cls.fit_transform(x_train)
# print(result.shape)
# cls.fit(x_train, y_train)
y_predict = grid_search.predict(x_test)
#ValueError: Found unknown categories ['Mississippi', 'New Hampshire', 'New Mexico', 'Edmonton'] in column 0 during transform
# This is because in train test doesn't have these value, these categories have few sample and when we split, all of them go to test set
# use handle_unknown

print(classification_report(y_test, y_predict))
#                                         precision    recall  f1-score   support

#                               accuracy                           0.63      1615
#                              macro avg       0.38      0.39      0.38      1615
#                           weighted avg       0.63      0.63      0.63      1615

# optimization

# the description (des_ft) is the main factor causing lowing the performance
# how to improve the script? Although tf/idf scored by the importance of tokens,
# but the machine still spend a little concern on the unimportance tokens => we should do sth to navigate the machine ignore all unimportance tokens
# => use min_df, max_df,   min_df=0.01, max_df=0.95 for des_ft

#      Decision tree                          precision    recall  f1-score   support
#
#                               accuracy                           0.63      1615
#                              macro avg       0.29      0.28      0.28      1615
#                           weighted avg       0.62      0.63      0.63      1615
# features reduce approximate 1000 times but performance still at the same for this decision tree model

# => use grid_search to find which model, hyperparameter is better
# => this features remained are equally important => use SelectKbest
# chi2
# Chi-squared stats of non-negative features for classification tasks.
# with ,in_df, max_df and selectKBest for k = 800 ft and model Random forest
#                                         precision    recall  f1-score   support
#                               accuracy                           0.75      1615
#                              macro avg       0.50      0.30      0.31      1615
#                           weighted avg       0.73      0.75      0.71      1615
# => better performance
# with ,in_df, max_df and selectKBest for k = 500 ft and model Random forest
#                                         precision    recall  f1-score   support
#                               accuracy                           0.76      1615
#                              macro avg       0.52      0.31      0.32      1615
#                           weighted avg       0.75      0.76      0.72      1615
# with ,in_df, max_df and selectKBest for 300 ft and model Random forest
#                                         precision    recall  f1-score   support
#                               accuracy                           0.76      1615
#                              macro avg       0.51      0.32      0.33      1615
#                           weighted avg       0.74      0.76      0.73      1615
# with ,in_df, max_df and selectKBest for 200 ft and model Random forest
#                                         precision    recall  f1-score   support
#                               accuracy                           0.73      1615
#                              macro avg       0.45      0.36      0.38      1615
#                           weighted avg       0.72      0.73      0.72      1615
# => become worse
# instead of using SelectKBest, use SelectPercentile
# Like we mentioned above the data is not balanced => balance data => can use over sampling
# if we use SMOTE, originally SMOTE just use for integer, so we use another version of this, use SMOTENC, SMOTEN
# In addition, we can group classes into one, solve each layers, but it has problems,
# in this case, group director_business_unit_leader, specialist, managing_director_small_medium_company into a class, so we have total 4 classes
# In ML, the model is already fast but we care about the performance. The problem is the difference of level, if the level of class is not adjacent

# grid search
