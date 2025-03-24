import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def create_ts_data(data, window_size = 5):
    i = 1
    while i < window_size:
        data['co2_{}'.format(i)] = data['co2'].shift(-i)
        i += 1
    data['target'] = data['co2'].shift(-i)
    data = data.dropna(axis=0)
    return data
data = pd.read_csv('co2.csv')
# visualization
# print(data.dtypes) # time often is object type
# Ox = time, Oy = co2
# change object type to time type
# pandas.to_datetime
data['time'] = pd.to_datetime(data['time'])
data['co2'] = data['co2'].interpolate()
data = create_ts_data(data)
# print(data.dtypes) # time often is object type
# fig, ax = plt.subplots()
# ax.plot(data['time'], data['co2'])
# ax.set_xlabel('Year')
# ax.set_ylabel('Co2')
# plt.show()

# solve Nan
# print(data.info())
print(data.isna().sum())
# time     0
# co2     59
# with time-series, we can not drop data, it interrupts data => look at data visualization, we cannot use median or avg
# use  interpolation  data['co2'] = data['co2'].interpolate()

# change time-series to the type that machine can know // create_ts_data
# by that sliding window, there are some Nan created, we defined window as 5 so we can drop it, also they are adjacent values

# now we split data
x = data.drop(['time', 'target'], axis=1)
y = data['target']
# if we use train_test_split, data would not be continuous(time series is continuous), no random if  we want to use train_test_split
train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples*train_ratio)]
y_train = y[:int(num_samples*train_ratio)]
x_test = x[int(num_samples*train_ratio):]
y_test = y[int(num_samples*train_ratio):]

reg = LinearRegression()
reg.fit(x_train, y_train)
# there is one line of code is redundant is ("scaler", StandardScaler())
y_predict = reg.predict(x_test)
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))

# Linear Regression
# MAE: 0.3605603788359202
# MSE: 0.2204494736034635
# R2: 0.9907505918201437

# we base on which metrics to predict this problem? this is coefficient of determination (R2)
# because in practically, mse and mae are hard to predict whether model is good or not
# Example: buy house, error about 5million => still buy
# But if we buy beef, error about 100k => not buy
# => mse, mae do not provide the ratio of actual price and predict

# Random forest:
# MAE: 5.647574561403413
# MSE: 49.939171364033704
# R2: -1.0952999912379058
# It's worse than Linear because
fig, ax = plt.subplots()
ax.plot(data['time'][:int(num_samples*train_ratio)], data['co2'][:int(num_samples*train_ratio)], label='train')
ax.plot(data['time'][int(num_samples*train_ratio):], data['co2'][int(num_samples*train_ratio):], label='test')
ax.plot(data['time'][int(num_samples*train_ratio):], y_predict, label='prediction')
ax.set_xlabel('Year')
ax.set_ylabel('Co2')
ax.legend()
ax.grid()
plt.show()
# visualize the result, we know Random forest is the collection of decision trees,
# in classification we have root node, but what is used for regression? each tree has their own result => avg them
# then we can see the plot, it shows that the train set in range of 310 to 360, so when it predicts, maximum value is 360.
# But the test set have values above the maximum so the model is worse than Linear
# Back to Linear, it is a straight line, so as long as it is on the line, they can predict

# this is fake data for 5 weeks to predict the week 6
current_data = [380.5, 390, 390.2, 390.4, 393]
# prediction = reg.predict([current_data]) # have to fit 2d matrix
# print(prediction)
# I want to use this data to predict not just week 6, but also week 7,8,9,..?? =>
for i in range(10):
    print("Input is {}".format(current_data))
    prediction = reg.predict([current_data])[0]
    print("CO2 in week {} is {}".format(i + 1, prediction))
    current_data = current_data[1:] + [prediction]
    print("-------------------------------------------")

# Trend: this decreases over the time
# R2: [0.9906941835498279, 0.9823771406805402, 0.9728273214391031]
# MSE: [0.22101469765793644, 0.4191794797112043, 0.647630915261317]
# MAE: [0.3618741479836254, 0.5058913973664143, 0.6465162623791926]