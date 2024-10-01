import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
# ......
# --- end of task --- #

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
data = np.loadtxt('data/crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]


# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = 0.75   
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 8 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
    # now train your model using (sample_train, label_train)
    model.fit(sample_train, label_train)
    # now evaluate your training error (MSE) and stores it in "er_train"
    predictions_train = model.predict(sample_train)
    er_train = mean_squared_error(label_train, predictions_train)

    er_train_alpha.append(er_train)
    # now evaluate your testing error (MSE) and stores it in "er_test"
    predictions_test = model.predict(sample_test)
    er_test = mean_squared_error(label_test, predictions_test)
    er_test_alpha.append(er_test)
    # --- end of task --- #

    
plt.plot(alpha_vec,er_train_alpha, label='Training Error')
plt.plot(alpha_vec,er_test_alpha, label='Testing Error')
plt.xlabel('Hyper-Parameter Alpha')
plt.ylabel('Prediction Error (MSE)')
plt.legend()
plt.show()