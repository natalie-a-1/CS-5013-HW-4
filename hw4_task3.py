import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
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
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.001, 0.1, 1, 10, 1000]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []
er_valid_alpha = []

k = 5
fold_size = num_train // k

for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    er_valid = 0
    
    # --- Your Task --- #
    # now implement k-fold cross validation 
    # on the training set (which means splitting 
    # training set into k-folds) to get the 
    # validation error for each candidate alpha value 
    # store it in "er_valid"
    for i in range(k):
            # Split the data into training and validation sets
            start = i * fold_size
            end = start + fold_size
            val_sample = sample_train[start:end]
            val_label = label_train[start:end]
            
            train_sample = np.concatenate((sample_train[:start], sample_train[end:]), axis=0)
            train_label = np.concatenate((label_train[:start], label_train[end:]), axis=0)
            
            # Train the model on the training set
            model.fit(train_sample, train_label)
            
            # Evaluate the model on the validation set
            val_predictions = model.predict(val_sample)
            er_valid += mean_squared_error(val_label, val_predictions)
        
        # Average validation error for this alpha
    er_valid /= k
    er_valid_alpha.append(er_valid)
    # --- end of task --- #


# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values
print("Validation Errors for each alpha value:")
for alpha, error in zip(alpha_vec, er_valid_alpha):
    print(f"Alpha: {alpha}, Validation Error: {error}")

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"
alpha_opt = alpha_vec[np.argmin(er_valid_alpha)]

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha=alpha_opt)
model.fit(sample_train, label_train)

predictions_train = model.predict(sample_train)
er_train = mean_squared_error(label_train, predictions_train)

# Evaluate the model on the testing set
predictions_test = model.predict(sample_test)
er_test = mean_squared_error(label_test, predictions_test)

print(f"Optimal Alpha: {alpha_opt}")
print(f"Training Error: {er_train}")
print(f"Testing Error: {er_test}")