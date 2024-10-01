import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample
# --- end of task --- #

# load an imbalanced data set 
# there are 50 positive class instances 
# there are 500 negative class instances 
data = np.loadtxt('data/diabetes_new.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# vary the percentage of data for training
num_train_per = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

# Hyper-parameter values for class_weight
class_weight_values = [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 5}, {0: 1, 1: 10}]
auc_hyper_param = []

for per in num_train_per: 

    # create training data and label
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]

    model = LogisticRegression(max_iter=1000)

    # --- Your Task --- #
    # Implement a baseline method that standardly trains 
    # the model using sample_train and label_train
    model.fit(sample_train, label_train)
    
    # evaluate model testing accuracy and stores it in "acc_base"
    predictions_test = model.predict(sample_test)
    acc_base = accuracy_score(label_test, predictions_test)
    acc_base_per.append(acc_base)
    
    # evaluate model testing AUC score and stores it in "auc_base"
    prob_test = model.predict_proba(sample_test)[:, 1]
    auc_base = roc_auc_score(label_test, prob_test)

    auc_base_per.append(auc_base)
    # --- end of task --- #
    
    
    # --- Your Task --- #
    # Now, implement your method 
    # Aim to improve AUC score of baseline 
    # while maintaining accuracy as much as possible 
    pos_samples = sample_train[label_train == 1]
    pos_labels = label_train[label_train == 1]
    neg_samples = sample_train[label_train == 0]
    neg_labels = label_train[label_train == 0]
    
    pos_samples_resampled, pos_labels_resampled = resample(pos_samples, pos_labels, 
                                                           replace=True, 
                                                           n_samples=len(neg_labels), 
                                                           random_state=42)
    
    sample_train_balanced = np.vstack((neg_samples, pos_samples_resampled))
    label_train_balanced = np.hstack((neg_labels, pos_labels_resampled))
    
    model.fit(sample_train_balanced, label_train_balanced)
    
    # evaluate model testing accuracy and stores it in "acc_yours"
    predictions_test = model.predict(sample_test)
    acc_yours = accuracy_score(label_test, predictions_test)
    acc_yours_per.append(acc_yours)
    
    # evaluate model testing AUC score and stores it in "auc_yours"
    prob_test = model.predict_proba(sample_test)[:, 1]
    auc_yours = roc_auc_score(label_test, prob_test)
    auc_yours_per.append(auc_yours)
    # --- end of task --- #

# Evaluate the impact of the hyper-parameter on AUC score
for class_weight in class_weight_values:
    model = LogisticRegression(max_iter=1000, class_weight=class_weight)
    model.fit(sample_train_balanced, label_train_balanced)
    prob_test = model.predict_proba(sample_test)[:, 1]
    auc = roc_auc_score(label_test, prob_test)
    auc_hyper_param.append(auc)

plt.figure()    
plt.plot(num_train_per,acc_base_per, label='Base Accuracy')
plt.plot(num_train_per,acc_yours_per, label='Your Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.title('Model Accuracy vs Training Data Size')

plt.figure()
plt.plot(num_train_per,auc_base_per, label='Base AUC Score')
plt.plot(num_train_per,auc_yours_per, label='Your AUC Score')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.legend()
plt.title('Model AUC Score vs Training Data Size')

# Plot the impact of the hyper-parameter on AUC score
plt.figure()
plt.plot([str(cw) for cw in class_weight_values], auc_hyper_param, marker='o')
plt.xlabel('Class Weight')
plt.ylabel('AUC Score')
plt.title('Impact of Class Weight on AUC Score')
plt.show()