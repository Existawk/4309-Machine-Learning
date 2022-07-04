#Parker Skinner
#1001541467
#4309-001
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Parker Skinner
# 1001541467


# In[ ]:


import numpy as np
import sys


# In[ ]:


def naive_bayes(training_file, test_file):
    training_data = np.loadtxt(training_file).tolist()
    test_data = np.loadtxt(test_file).tolist()
    
    training_dict = {}
    training_dict_results= {}
    
    for i in range(len(training_data)):
        if training_data[i][-1] in training_dict:
            training_dict[training_data[i][-1]].append(training_data[i][:-1])
        else:
            training_dict[training_data[i][-1]] = [training_data[i][:-1]]
           
    for i in training_dict:
        for j in range(len(training_dict[i])):
            for k in training_dict[i]:
                sum += training_dict[i][k][j]


# In[ ]:


naive_bayes(sys.argv[1],sys.argv[2])


# %%
