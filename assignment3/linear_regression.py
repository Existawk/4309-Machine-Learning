# Parker Skinner
# 1001541467
# CSE 4309-001
# 9/26/2021

import numpy as np
import sys

def phi(vector, degree):
    temp = [1]
    
    for i in range(len(vector)):
        for k in range(1, degree+1):
            temp.append(vector[i]**k)
    
    return(temp)

# MAIN
training_data = np.loadtxt(sys.argv[1]).tolist()
test_data = np.loadtxt(sys.argv[2]).tolist()
degree = int(sys.argv[3])
lambda_num = int(sys.argv[4])

training_list = []
classes = []

for i in range(len(training_data)):
    vector = phi(training_data[i][:-1], degree)
    print(vector)
    training_list.append(vector)
    classes.append(training_data[i][-1])

weights = np.dot(np.dot(np.linalg.pinv((np.dot(lambda_num,np.identity(len(training_list[0]))) + np.dot(np.transpose(training_list), training_list))),np.transpose(training_list)),classes)

for i in range(len(weights)):
    print("w{:d}={:.4f}".format(i, weights[i]))

for i in range(len(test_data)):
    output = np.dot(np.transpose(weights), phi(test_data[i][:-1], degree))
    target = test_data[i][-1]
    error = (target-output)**2
    print("ID={:5d}, output={:14.4f}, target value = {:10.4f}, squared error = {:0.4f}".format(i+1,output,target,error))
