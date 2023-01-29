import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import matplotlib.pyplot as plt

filename1 = r'..\data\H\pa_h'
filename2 = r'..\data\ep_h_l'
feature_name = ['L/P', 'L', 't1', 't2']
X = np.genfromtxt(filename1 + '.csv', delimiter=',')[:, :-1]
a = X.copy()
a[:, 0] = X[:, 1] / X[:, 0]
X = a
y = np.genfromtxt(filename2 + '.csv', delimiter=',')[:, 1]
y1 = np.zeros((y.shape))
y1[y < 0.2] = 1
y = y1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
print(X_train.shape, y_train.shape)
dt_reg = DecisionTreeClassifier(
    max_depth=1
    , min_samples_split=30
    , min_samples_leaf=30
)
dt_reg.fit(X_train, y_train)
import graphviz

dot_data = sklearn.tree.export_graphviz(dt_reg, None
                                        , feature_names=feature_name
                                        )
graph = graphviz.Source(dot_data)
graph.view()
print(dt_reg.score(X_train, y_train))
print(dt_reg.score(X_test, y_test))
print(dt_reg.feature_importances_)
# # print(dt_reg.predict(X_test[0].reshape(1,-1)))
