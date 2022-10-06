import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
red_wine = pd.read_csv('./winequality-red.csv', sep=';')
white_wine = pd.read_csv('./winequality-white.csv',sep=';')
df = pd.concat([red_wine, white_wine])
df.reset_index(drop=True, inplace=True)
df.info()
df.head() # To get first n rows from the dataset default value of n is 5
####Ktfgfgdfgdfg

y = df['quality'] # 레이블(종속변수)
X = df.drop(['quality'], axis=1, inplace=False) # 피처(독립변수)
X= X.to_numpy()
X1 = X
#X = df.values[:, 0:11]  # get input values from first two columns
#y = df.values[:, 0]  # get output values from last coulmn
m = len(y) # Number of training examples

print('Total no of training examples (m) = %s \n' %(m))
print(X[0:1,:])
# Show only first 5 records
# for i in range(5):
#     print('x =', X[i], ', y =', y[i])

def feature_normalize(X):
  """
    Normalizes the features(input variables) in X.

    Parameters
    ----------
    X : n dimensional array (matrix), shape (n_samples, n_features)
        Features(input varibale) to be normalized.

    Returns
    -------
    X_norm : n dimensional array (matrix), shape (n_samples, n_features)
        A normalized version of X.
    mu : n dimensional array (matrix), shape (n_features,)
        The mean value.
    sigma : n dimensional array (matrix), shape (n_features,)
        The standard deviation.
  """
  #Note here we need mean of indivdual column here, hence axis = 0
  mu = np.mean(X, axis = 0)  
  # Notice the parameter ddof (Delta Degrees of Freedom)  value is 1
  sigma = np.std(X, axis= 0, ddof = 1)  # Standard deviation (can also use range) 표준편차 ddof를 지정해주는 이유는 numpy의 std default가 0이기 때문
  X_norm = (X - mu)/sigma
  return X_norm, mu, sigma

X, mu, sigma = feature_normalize(X)

print('mu= ', mu)
print('sigma= ', sigma)
print('X_norm= ', X[:5])

mu_testing = np.mean(X, axis = 0) # mean
mu_testing

sigma_testing = np.std(X, axis = 0, ddof = 1) # mean
sigma_testing

# Lets use hstack() function from numpy to add column of ones to X feature 
# This will be our final X matrix (feature matrix)
X = np.hstack((np.ones((m,1)), X))
X[:5]