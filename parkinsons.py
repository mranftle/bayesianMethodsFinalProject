import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn import linear_model
import plotly.plotly as py
import plotly.graph_objs as go

data1 = pd.read_csv("data/parkinsons_data.csv")
print len(data1.columns)
data1 = data1.drop('age',axis = 1)
data1 = data1.drop('sex',axis = 1)
data1 = data1.drop('test_time',axis = 1)

train = data1.loc[data1['subject#'] <=32]
test = data1.loc[data1['subject#'] > 32]

y_motor = train['motor_UPDRS']
y_total = train['total_UPDRS']
X = train.drop('motor_UPDRS',axis = 1)
X = X.drop('total_UPDRS',axis = 1)


labels_motor = test['motor_UPDRS']
labels_total = test['total_UPDRS']
test = test.drop('motor_UPDRS',axis = 1)
test = test.drop('total_UPDRS',axis = 1)

print "Gaussian Process"
kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp.fit(X, y_total)
GaussianProcessRegressor(alpha=1e-10, copy_X_train=True,
kernel=1**2 + Matern(length_scale=2, nu=1.5) + WhiteKernel(noise_level=1),
n_restarts_optimizer=0, normalize_y=False,
optimizer='fmin_l_bfgs_b', random_state=None)

print "Linear Regression:"
clf = LinearRegression()
clf.fit(X,y_total)
y_pred = clf.predict(test)
print "Rsquared value:"
print clf.score(test,labels_total)
print "Ceffecients:"
print clf.coef_
print "Intercept:"
print clf.intercept_
print "Mean Squared Error:"
print mean_squared_error(labels_total, y_pred)
print "Percent Correct:"
print "\n"


print "Decision Tree Total UPDRS"
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf.fit(X,y_total)
y_pred = clf.predict(test)
print "Rsquared value:"
print clf.score(test,labels_total)
print "Mean Squared Error:"
print mean_squared_error(labels_total, y_pred)
print '\n'

print "Decision Tree Motor UPDRS"
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf.fit(X,y_motor)
y_pred = clf.predict(test)
print "Rsquared value:"
print clf.score(test,labels_motor)
print "Mean Squared Error:"
print mean_squared_error(labels_motor, y_pred)
print '\n'

# py.sign_in('Mranftle', 'MikQahEQpuPow1Dd5XxZ')
# trace1 = go.Scatter(
#     x = [x for x in range(0, len(labels_total))],
#     y = labels_total,
#     mode = 'lines',
#     name = 'True Values'
# )
# trace2 = go.Scatter(
#     x = [x for x in range(0,len(labels_total))],
#     y = y_pred,
#     mode = 'lines',
#     name = 'Predicted Values'
# )
#
# data = [trace1, trace2]
# py.iplot(data, filename='DecisionTreeRegressor')

#
# print "Exponential"
# clf = LinearRegression()
# y =np.log(y)
# clf.fit(X,y)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels)
# print "Ceffecients:"
# print clf.coef_
# print "Intercept:"
# print clf.intercept_
# print "Mean Squared Error:"
# print mean_squared_error(labels, y_pred)
# print "\n"
