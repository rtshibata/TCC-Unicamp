import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

#Reading data
data_matrix = pd.read_csv("/home/ec2010/ra082674/TCC/TCC_RenatoShibata/base")

data_matrix.insert(0, "x0", 1)
numpy_data = data_matrix.as_matrix()
ys = numpy_data[:,59]
ys = ys.reshape(31715,1)
numpy_data = np.delete(numpy_data, [59], axis=1)

k, bins, patches = plt.hist(ys, bins = 10, range = (-1,2001))
plt.xlabel('Number of Shares', fontsize=15)
plt.ylabel('Number of Ocurrences', fontsize=15)

plt.show()
print("Total number of shares in the bins (0-10000 shares): " + repr( k.sum()))

index = np.argwhere(ys > 2000)
index = np.delete(index, [1], axis=1)
   
ys = np.delete(ys, index, axis = 0)
numpy_data = np.delete(numpy_data, index, axis=0)

#Normalization
maximum_features = np.zeros([1, 59])
minimum_features = np.zeros([1,59])
mean_features = np.zeros([1,59])
interval_features = np.zeros([1,59])

for i in range(0,59):
     maximum_features[0,i] = numpy_data[:,i].max()
     
for i in range(0,59):
     minimum_features[0,i] = numpy_data[:,i].min()
     
for i in range(0,59):
     mean_features[0,i] = numpy_data[:,i].mean()
     
for i in range(0,59):
     interval_features[0,i] = maximum_features[0,i] -  minimum_features[0,i]
 

list_to_normalize = [1,2,3,4,5,6,7,8,9,10,11,18,19,20,21,22,23,24,25,26,27,28,29,
                     38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58]
                     
    
for i in list_to_normalize:
     numpy_data[:,i] = (numpy_data[:,i] - mean_features[0,i])/(interval_features[0,i])
    
 
#Cost function computation and Gradient Descent with regularization
thetas = np.zeros([59,1])

Lambda = 150

m = ys.size

iterations = 1000

alpha = 1.3


def compute_cost(numpy_data, ys, thetas, m, Lambda):

       
    thetas_squared = thetas ** 2
    predictions = numpy_data.dot(thetas)
    sqErrors = (predictions - ys) ** 2
    J = (1.0 / (2 * m)) * (sqErrors.sum() + Lambda*thetas_squared.sum() - Lambda*thetas_squared[0,0])
    
    
    return J


def gradient_descent(numpy_data, ys, thetas, alpha, iterations, m, Lambda):

    J_history = np.zeros([iterations, 1])
    
    tempthetas = np.zeros([thetas.size, 1])

    for i in range(iterations):
        predictions = numpy_data.dot(thetas)
        errors = predictions - ys
        tempthetas = thetas - (alpha/m)*(numpy_data.transpose().dot(errors))
        for j in range(1,59):
            tempthetas[j,0] = tempthetas[j,0] - ((alpha*Lambda)/m)*thetas[j,0]
        thetas = tempthetas
        J_history[i, 0] = compute_cost(numpy_data, ys, thetas, m, Lambda)
    return thetas, J_history, i

start_time = time.clock()

thetas, J_history, i = gradient_descent(numpy_data, ys, thetas, alpha, iterations, m, Lambda)

print("Lambda: "+ repr(Lambda))
print("Alpha: " + repr(alpha))
print ("Time (in seconds): " + repr((time.clock() - start_time)))
print("Final Cost Function: " + str((J_history[iterations-1,0])))


iterations_vector = np.array(range(0,iterations))

plt.figure(figsize=(2.5, 2.5), dpi=75)
plt.scatter(iterations_vector, J_history, s = 10)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Cost Function', fontsize=15)

plt.show()

#Reading the test set and the target values for the test set
test_matrix = pd.read_csv("/Users/FelipeMoret/Downloads/mo444-assignment-01.zip/test.csv")
test_matrix.drop(test_matrix.columns[[0 , 1]], axis=1, inplace = True)
test_matrix.insert(0, "x0", 1)
numpy_test = test_matrix.as_matrix()

#Normalization of test set
maximum_features_test = np.zeros([1, 59])
minimum_features_test = np.zeros([1,59])
mean_features_test = np.zeros([1,59])
interval_features_test = np.zeros([1,59])


for i in range(0,59):
     maximum_features_test[0,i] = numpy_test[:,i].max()
     
for i in range(0,59):
     minimum_features_test[0,i] = numpy_test[:,i].min()
     
for i in range(0,59):
     mean_features_test[0,i] = numpy_test[:,i].mean()
     
for i in range(0,59):
     interval_features_test[0,i] = maximum_features_test[0,i] -  minimum_features_test[0,i]

for i in list_to_normalize:
     numpy_test[:,i] = (numpy_test[:,i] - mean_features_test[0,i])/(interval_features_test[0,i])
    

test_results = pd.read_csv("/Users/FelipeMoret/Downloads/mo444-assignment-01.zip/test_target.csv")
numpy_results = test_results.as_matrix()


#Evaluating predictions for the test set
predictions_for_test = numpy_test.dot(thetas)


n = numpy_results.size

#Evaluating the error in predictions
error = predictions_for_test - numpy_results


error = np.absolute(error)

#Evaluating MAE

MAE =  (1.0/n)*(error.sum())
print("MAE: " + repr(MAE))

