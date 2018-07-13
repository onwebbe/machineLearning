import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import image
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston_housing_price = datasets.load_boston()
# print(boston_housing_price)
X_full = boston_housing_price.data
Y = boston_housing_price.target
boston = pd.DataFrame(X_full, columns = boston_housing_price.feature_names)
boston["PRICE"] = Y
boston.head()

boston = boston.dropna(axis = 0)
for col in np.take(boston.columns, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    boston[col] /= boston[col].max()
train, test = train_test_split(boston.copy(), test_size = 0.5)

print (boston_housing_price.feature_names)
# features data will be x1, x2, x3, ... and parameters will be w1, w2, w3, ...
# so we expect that the price will be impacted by features and parameters
# so predict price will be h(x) = x1 * w1 + x2 * w2 + x3 * w3 + ... + b

def linear(features, pars) :
    price = np.sum(features * pars[:-1], axis = 1) + pars[-1]
    return price

def error_squre(predict_value, value):
    return sum(np.array((predict_value - value) ** 2))

def cost(df, features, params):
    df['predict'] = linear(df[features].values, params)
    cost = error_squre(df['predict'], df['PRICE']) / len(df)
    return cost

def gradient(training, features, parameters):
    Gradient = np.zeros(len(parameters))
    for i in range(len(parameters)):
        parameters_new = parameters.copy()
        parameters_new[i] += 0.01
        Gradient[i] = (cost(training, features, parameters_new) - cost(training, features, parameters)) / 0.01
    return Gradient



# print ("--------------------------   1   --------------------------")
# print (cost(train, ['CRIM', 'AGE', 'DIS'], [1, 1, 1, 5]))
# print (gradient(train, ['CRIM', 'AGE', 'DIS'], [1, 1, 1, 5]))

def gradientDecent(training, loopingCount, stepLength, features, parameters):
    for i in range(loopingCount):
        gradientValue = gradient(training, features, parameters)
        gradientValue = gradientValue * stepLength
        parameters = parameters - gradientValue
        if (i % 50 == 0):
            print ("--------------------------   ", i, "   --------------------------")
            print (parameters)
            print (cost(training, features, parameters))
    return parameters


features_using = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
# run the training using below line
# trainedParameters = gradientDecent(train, 10000, 0.1, features_using, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# print (trainedParameters)


# after get the training result, just using the result as below. Using the 50% boston data to train and iterate 10000 by step length 0.1
trainedParameters = [-10.11286235, 6.18640666, 2.65383433, 1.13253285, -15.88749602
, 28.21887833, 1.54930346, -16.8741565, 7.11506397, -9.24856385
, -22.87412843, 4.00703957, -21.81559481, 40.57212234]

boston['cal_predict'] = linear(boston[features_using].values, trainedParameters)
print (mean_squared_error(boston['PRICE'], boston['cal_predict']))
plt.scatter(boston.cal_predict, boston.PRICE)
plt.xlabel("cal_predict")
plt.ylabel("price")
plt.show()
print (boston)
# the_predict_result = error_squre(boston['predict'], boston['PRICE'])
