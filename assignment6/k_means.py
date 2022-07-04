# Parker Skinner
# 1001541467
# CSE 4309-001
# 11/11/2021

import numpy as np
import sys
import random
import math

def set_clusters(data, K):
    cluster = {}
    for i in range(K):
        cluster[i] = []
    for i in range(len(data)):
        cluster[data[i][1]].append(data[i][0])
    return cluster

def get_cluster_mean(cluster, K):
    cluster_mean = []
    for i in range(K):
        if cluster[i]:
            cluster_mean.append(np.mean(cluster[i],axis=0))
        else:
            cluster_mean.append(0)
    return cluster_mean

def distance(value, mean, dimensions):
    if dimensions == 1:
        return abs(mean - value)
    else:
        return math.sqrt(((mean[0]-value[0])**2)+((mean[1]-value[1])**2))

# MAIN

#load data
data_file = np.loadtxt(sys.argv[1]).tolist()
K = int(sys.argv[2])
initialization = sys.argv[3]
dimensions = 0

#find dimension
if (type(data_file[0]) == float):
    dimensions = 1
else:
    dimensions = 2

data = []
for i in data_file:
    data.append([i,-1])

#set round robin initial clusters
if (initialization == "round_robin"):
    counter = 0
    for i in range(len(data)):
        data[i][1] = counter
        counter += 1
        if counter > K-1:
            counter = 0
        
#set random clusters
if (initialization == "random"):
    for i in range(len(data)):
        data[i][1] = random.randint(0, K-1)

#group data by cluster and find means
cluster = set_clusters(data, K)
cluster_mean = get_cluster_mean(cluster, K)
n = 1

while(n):
    for i in range(len(data)):
        clust = -1 
        min_distance = 999999999
        for j in range(len(cluster_mean)):
            if (distance(data[i][0], cluster_mean[j], dimensions) < min_distance):
                clust = j
                min_distance = distance(data[i][0], cluster_mean[j], dimensions)
        data[i][1] = clust

    new_cluster = set_clusters(data,K)
    cluster_mean = get_cluster_mean(new_cluster, K)
    if(cluster == new_cluster):
        n = 0
    cluster = new_cluster

if (dimensions == 1):
    for i in range(len(data_file)):
        print("{:10.4f} --> cluster {:d}".format(data[i][0],  data[i][1]+1))
else:
    for i in range(len(data_file)):
        print("({:10.4f}, {:10.4f}) --> cluster {:d}".format(data[i][0][0], data[i][0][1], data[i][1]+1))