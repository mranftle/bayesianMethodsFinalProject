import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import fmin_tnc
from sklearn.metrics import mean_squared_error
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RationalQuadratic
from sklearn.linear_model import BayesianRidge
import plotly.plotly as py
import plotly.graph_objs as go
import GPy
from IPython.display import display
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

data1 = pd.read_csv("data/parkinsons_data.csv")
data1 = data1.drop('age',axis = 1)
data1 = data1.drop('sex',axis = 1)
data1 = data1.drop('test_time',axis = 1)
print data1.columns
train = data1.loc[data1['subject#'] <=32].sample(n=1000)
test = data1.loc[data1['subject#'] > 32].sample(n=100)

y_motor = np.array(train['motor_UPDRS'], ndmin=2).transpose()
y_total = np.array(train['total_UPDRS'], ndmin=2).transpose()
X = train.drop('motor_UPDRS',axis = 1)
X = X.drop('total_UPDRS',axis = 1)
X = X.drop('subject#', axis = 1)
X = X.as_matrix()

labels_motor = np.array(test['motor_UPDRS'], ndmin=2).transpose()
labels_total = np.array(test['total_UPDRS'], ndmin=2).transpose()
test = test.drop('motor_UPDRS',axis = 1)
test = test.drop('total_UPDRS',axis = 1)
test = test.drop('subject#', axis=1)
test = test.as_matrix()

print y_total.shape
# kernel1 = GPy.kern.Linear(input_dim=16) GPy.kern.RBF(input_dim =16, variance=1., lengthscale=1.)
kernel1 =  GPy.kern.Linear(input_dim =16, variances=1)
kernel2 = GPy.kern.RBF(input_dim =16, variance=1., lengthscale=1.)
kernel3 = GPy.kern.Poly(input_dim=16, variance=1, scale=1)
kernel4 = GPy.kern.Matern32(input_dim=16, variance=1)
kernel5 = GPy.kern.RatQuad(input_dim=16, variance=1)
kernel6 = GPy.kern.White(input_dim=16)
# kernel = kernel5 + (kernel4*kernel2)
# kernel = kernel2 + (kernel6*kernel4)
kernel = kernel4 + kernel5 + kernel2*kernel1
m = GPy.models.GPRegression(X,y_total, kernel)
m.optimize(messages=True, optimizer='lbfgs') #lbfgs
mean,var = m.predict(test)
print mean_squared_error(labels_total, mean)


# print "Gaussian Process Total"
# kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1) + RationalQuadratic(length_scale=1)
# # kernel = 26.4**2 + Matern(length_scale=1e-05, nu=1) + WhiteKernel(noise_level=103)
#
# gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4)
# gp.fit(X, y_total)
# y_pred, sigma = gp.predict(test, return_std=True)
#
# print mean_squared_error(labels_total, y_pred)

# print "Gaussian Process Motor"
# kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1) + RationalQuadratic(length_scale=1)
# # kernel = 26.4**2 + Matern(length_scale=1e-05, nu=1) + WhiteKernel(noise_level=103)
#
# gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4)
# gp.fit(X, y_motor)
# y_pred, sigma = gp.predict(test, return_std=True)
#
# print mean_squared_error(labels_motor, y_pred)

# print "Linear Regression Total"
# clf = LinearRegression()
# clf.fit(X,y_total)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels_total)
# print "Ceffecients:"
# print clf.coef_
# print "Intercept:"
# print clf.intercept_
# print "Mean Squared Error:"
# lrt_mse = round(mean_squared_error(labels_total, y_pred),2)
# print lrt_mse
# print "\n"
#
# print "Linear Regression Motor"
# clf = LinearRegression()
# clf.fit(X,y_motor)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels_motor)
# print "Ceffecients:"
# print clf.coef_
# print "Intercept:"
# print clf.intercept_
# print "Mean Squared Error:"
# lrm_mse = round(mean_squared_error(labels_motor, y_pred),2)
# print lrm_mse
# print "\n"\
#
# print "Decision Tree Total UPDRS"
# from sklearn import tree
# clf = tree.DecisionTreeRegressor()
# clf.fit(X,y_total)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels_total)
# print "Mean Squared Error:"
# dtt_mse = round(mean_squared_error(labels_total, y_pred),2)
# print dtt_mse
# print '\n'
#
# print "Decision Tree Motor UPDRS"
# from sklearn import tree
# clf = tree.DecisionTreeRegressor()
# clf.fit(X,y_motor)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels_motor)
# print "Mean Squared Error:"
# dtm_mse = round(mean_squared_error(labels_motor, y_pred),2)
# print dtm_mse
# print '\n'
#
# print "Linear Regression Exponential Total"
# clf = LinearRegression()
# y =np.log(y_total)
# clf.fit(X,y)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels_total)
# print "Ceffecients:"
# print clf.coef_
# print "Intercept:"
# print clf.intercept_
# print "Mean Squared Error:"
# lret_mse = round(mean_squared_error(labels_total, y_pred),2)
# print lret_mse
# print "\n"
#
# print "Linear Regression Exponential Motor"
# clf = LinearRegression()
# y =np.log(y_motor)
# clf.fit(X,y)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels_motor)
# print "Ceffecients:"
# print clf.coef_
# print "Intercept:"
# print clf.intercept_
# print "Mean Squared Error:"
# lrem_mse= round(mean_squared_error(labels_motor, y_pred),2)
# print lrem_mse
# print "\n"
#
# print "Bayesian Ridge Regression Total"
# clf = BayesianRidge(compute_score=True)
# clf.fit(X,y_total)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels_total)
# print "Ceffecients:"
# print clf.coef_
# print "Intercept:"
# print clf.intercept_
# print "Mean Squared Error:"
# brt_mse= round(mean_squared_error(labels_total, y_pred),2)
# print brt_mse
# print "\n"
#
# print "Bayesian Ridge Regression Motor"
# clf = BayesianRidge(compute_score=True)
# clf.fit(X,y_motor)
# y_pred = clf.predict(test)
# print "Rsquared value:"
# print clf.score(test,labels_motor)
# print "Ceffecients:"
# print clf.coef_
# print "Intercept:"
# print clf.intercept_
# print "Mean Squared Error:"
# brm_mse = round(mean_squared_error(labels_motor, y_pred),2)
# print brm_mse
# print "\n"
#
py.sign_in('Mranftle', 'MikQahEQpuPow1Dd5XxZ')
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

# trace1 = go.Bar(
#     x=['RBF', 'Lin', 'Mat', 'RQ', 'Poly'],
#     y=[158.45, 164.70, 143.26, 187.90, 181.20],
#     name='LBFGS'
# )
# trace2 = go.Bar(
#     x=['RBF', 'Lin', 'Mat', 'RQ', 'Poly'],
#     y=[142.71, 181.53, 147.10, 160.95, 162.05],
#     name='TNC'
# )
# data = [trace1, trace2]
# layout = go.Layout(
#     barmode='group',
#     title='Kernels',
#     yaxis=dict(
#         title='Mean Squared Error'
#     )
#
#
# )
#
# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='kerns')
# data = [trace1, trace2]
# py.iplot(data, filename='DecisionTreeRegressor')