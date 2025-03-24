import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

data = pd.read_excel('diabetes_data.xlsx')
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file('clas_diabetes.html')

target = 'DiabeticClass'
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# preprocessing
num_transformer = Pipeline(steps=[
    ('scaler',  StandardScaler())
])

gender_values = x_train['Gender'].unique()
excessUrination_values = x_train['ExcessUrination'].unique()
polydipsia_values = x_train['Polydipsia'].unique()
weightLossSudden_values = x_train['WeightLossSudden'].unique()
fatigue_values = x_train['Fatigue'].unique()
polyphagia_values = x_train['Polyphagia'].unique()
genitalThrush_values = x_train['GenitalThrush'].unique()
blurredVision_values = x_train['BlurredVision'].unique()
itching_values = x_train['Itching'].unique()
irritability_values = x_train['Irritability'].unique()
delayHealing_values = x_train['DelayHealing'].unique()
partialPsoriasis_values = x_train['PartialPsoriasis'].unique()
muscleStiffness_values = x_train['MuscleStiffness'].unique()
alopecia_values = x_train['Alopecia'].unique()
obesity_values = x_train['Obesity'].unique()

ord_transformer = Pipeline(steps=[
    ('encoder',  OrdinalEncoder(categories=[
        gender_values, excessUrination_values,
        polydipsia_values, weightLossSudden_values,
        fatigue_values, polyphagia_values,
        genitalThrush_values, blurredVision_values,
        itching_values, irritability_values,
        delayHealing_values, partialPsoriasis_values,
        muscleStiffness_values, alopecia_values, obesity_values
    ]))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["Age"]),
    ("ord_feature", ord_transformer, ['Gender', 'ExcessUrination', 'Polydipsia',
                                      'WeightLossSudden', 'Fatigue', 'Polyphagia',
                                      'GenitalThrush', 'BlurredVision', 'Itching',
                                      'Irritability', 'DelayHealing', 'PartialPsoriasis',
                                      'MuscleStiffness', 'Alopecia', 'Obesity']),

])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200))
])
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))
# for i, j in zip(y_predict, y_test):
#     print('Prediction: {}. Actual value: {}'.format(i, j))

# params = {
#     "model__n_estimators": [100, 200, 300],
#     "model__criterion": ["gini", "entropy", "log_loss"]
# }
# #                                                          not bring to params, it just a parameter not hyper
# grid_search = GridSearchCV(estimator=cls, param_grid=params, cv=4, scoring='recall' , verbose=2) # cv: cross validation
# grid_search.fit(x_train, y_train) # after doing it, it just save the best record
# print(grid_search.best_score_) # in default, this score refer to accuracy, change it
# print(grid_search.best_params_)