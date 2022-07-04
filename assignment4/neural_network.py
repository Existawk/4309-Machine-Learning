# Parker Skinner
# 1001541467
# CSE 4309-001
# 10/10/2021

import numpy as np
import sys
import random
import math

def find_max(a):
    return max(a)

def learning_rate(round):
    return 0.98**(round)

def createJ(D,K,units_per_layer, layers):
    J = np.zeros(layers, dtype=int)
    for i in range(len(J)):
        J[i] = units_per_layer
    J[0] = D
    J[-1] = K
    return J

def createT(training_data, classes, K):
    T = np.zeros((len(training_data), K))
    for i in range(len(training_data)):
        T[i] = classes[(training_data[i][-1])]
    return T

# MAIN

#load data
training_data = np.loadtxt(sys.argv[1]).tolist()
test_data = np.loadtxt(sys.argv[2]).tolist()
layers = int(sys.argv[3])
units_per_layer = int(sys.argv[4])
num_rounds = int(sys.argv[5])

#parse data into data structures

training_list = []
classes = dict()
weights = []


for i in range(len(training_data)):
     if training_data[i][-1] not in classes:
         classes[training_data[i][-1]] = []
counter = 0
for i in sorted(classes.keys()):
    temp = np.zeros(len(classes))
    temp[counter] = 1
    counter +=1
    classes[i] = temp

current_max = 0
D = len(training_data[0])-1
K = len(classes)
maxDim =  max(units_per_layer,K,D)
J = createJ(D,K,units_per_layer,layers)
t = createT(training_data, classes, K)
b = np.zeros((layers,maxDim)) 
w = np.zeros((layers,maxDim,maxDim))

#init b weights

for i in range (len(b)):
    for j in range (len(b[i])):
        b[i][j] = random.uniform(-0.05,0.05)

#init w weights

for i in range (len(w)):
    for j in range (len(w[i])):
        for k in range (len(w[i][j])):
            w[i][j][k] = random.uniform(-0.05,0.05)

#Normalize data

for i in range(len(training_data)):
    temp_max = find_max(training_data[i][:-1])
    if  temp_max > current_max:
        current_max = temp_max

for i in range(len(training_data)):
    for k in range(len(training_data[i])-1):
        training_data[i][k] /= current_max

for i in range(len(test_data)):
    for k in range(len(test_data[i])-1):
        test_data[i][k] /= current_max
           

#Training

for current_round in range (num_rounds):
    print(current_round)
    for n in range (len(training_data)):   
        z = np.zeros(layers, dtype=np.ndarray)
        z[0] = training_data[n][:-1]

        a = np.ndarray(layers, dtype=np.ndarray)
        for L in range(1, layers):
            a[L] = np.zeros(J[L])
            z[L] = np.zeros(J[L])
            for i in range(J[L]):
                weightedsum = 0
                for j in range(J[L-1]):
                    weightedsum += (w[L][i][j] * z[L-1][j])
                a[L][i] = weightedsum + b[L][i]
                z[L][i] = 1/(1+math.exp(-a[L][i]))
        
        Delta = np.zeros(layers, dtype=np.ndarray)
        for i in range(layers):
            Delta[i] = np.zeros(J[i])
        for i in range(J[-1]):
            Delta[layers - 1][i] = (z[layers - 1][i] - t[n][i]) * z[layers - 1][i] * (1 - z[layers - 1][i])
        for L in range(layers-2,0,-1):
            Delta[L] = np.zeros(J[L])
            for i in range(J[L]):
                weightedsum = 0
                for k in range(J[L+1]):
                    weightedsum += Delta[L+1][k] * w[L+1][k][i]
                Delta[L][i] = weightedsum * z[L][i] * (1-z[L][i])

        for L in range(1, layers):
            for i in range(J[L]):
                b[L][i] = b[L][i] - learning_rate(current_round+1) * Delta[L][i]
                for j in range(J[L-1]):
                    w[L][i][j] = w[L][i][j] - (learning_rate(current_round+1) * Delta[L][i] * z[L-1][j])

#Classification
ClassificationAcc = 0
Test_data_t = createT(test_data, classes, K)
for n in range(len(test_data)):
    Accuracy = 0
    PredictedClass = []
    PredictedNum = 0

    z = np.zeros(layers, dtype=np.ndarray)
    z[0] = test_data[n][:-1]

    a = np.ndarray(layers, dtype=np.ndarray)
    for L in range(1, layers):
        a[L] = np.zeros(J[L])
        z[L] = np.zeros(J[L])
        for i in range(J[L]):
            weightedsum = 0
            for j in range(J[L-1]):
                weightedsum += (w[L][i][j] * z[L-1][j])
            a[L][i] = weightedsum + b[L][i]
            z[L][i] = 1/(1+math.exp(-a[L][i]))

    for k in range(J[-1]):
        if z[layers-1][k] == PredictedNum:
            PredictedClass.append((sorted(classes.keys())[k]))
        if z[layers-1][k] > PredictedNum:
            PredictedNum = z[layers-1][k]
            PredictedClass.clear()
            PredictedClass.append((sorted(classes.keys())[k]))
    if (test_data[n][-1]) in PredictedClass:
        Accuracy = 1
    else:
        Accuracy = 0
    ClassificationAcc += Accuracy
    print("ID={:5d}, predicted={:10s}, true={:10s}, accuracy={:4.2f}".format(n+1,str(PredictedClass[0]),str(test_data[n][-1]), Accuracy))
print("classification accuracy={:6.4f}".format(ClassificationAcc/len(test_data)))