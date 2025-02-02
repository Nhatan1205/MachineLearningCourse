import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")
sn.heatmap(data.corr(), annot=True)
plt.show()
# print(data)
