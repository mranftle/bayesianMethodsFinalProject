import pandas as pd
import numpy as np
import GPy

data1 = pd.read_csv("data/parkinsons_data_classification.csv")
# data1 = data1.drop('name', axis=1)
data1 = data1.drop('spread1',axis=1)
data1 = data1.drop('spread2', axis=1)
data1 = data1.drop('D2',axis=1)
data1 = data1.drop('RPDE',axis=1)

data1['name'] = data1['name'].apply(lambda x: int(str(x).split('_')[2][1:]))
train = data1.loc[data1['name'] <=32].sample(n=100)
test = data1.loc[data1['name'] > 32].sample(n=10)
X = train.drop('name',axis=1)
X = X.drop('status', axis=1)
test = test.drop('name',axis=1)

y = np.array(train['status'], ndmin=2).transpose()
y_star = np.array(test['status'], ndmin=2).transpose()
test = test.drop('status',axis=1)

# define kernels here
k = GPy.kern.RBF(1, variance=7., lengthscale=0.2)
kernel1 =  GPy.kern.Linear(input_dim =16, variances=1)
kernel2 = GPy.kern.RBF(input_dim =16, variance=1., lengthscale=1.)
kernel3 = GPy.kern.Poly(input_dim=16, variance=1, scale=1)
kernel4 = GPy.kern.Matern32(input_dim=16, variance=1)
kernel5 = GPy.kern.RatQuad(input_dim=16, variance=1)
kernel6 = GPy.kern.White(input_dim=16)

# k = linear kernel combinations 

X = X.as_matrix()
test = test.as_matrix()
gp = GPy.models.GPClassification(X,y)
gp.optimize(messages=True)
mean, var = gp.predict(test)

predictions = [m > 0.5 for m in mean]

compare = zip(y_star, predictions)
print compare
print sum([1 for x in compare if x[0] == x[1]])/float(len(compare))