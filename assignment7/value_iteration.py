# Parker Skinner
# 1001541467
# CSE 4309-001
# 12/01/2021

import numpy as np
import sys
import csv

#FUNCTIONS

directions = {"^","v","<",">"}

#Reward function
def R(x, y):
    data = environment_data[x][y]
    if data == 'X':
        return 0
    elif data == '.':
        return non_terminal_reward
    else:
        return float(data)

def transitions(Sprime, s , move):
    temp = 0
    if move == "^":
        nextState = (s[0]-1, s[1]) 
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.8
        nextState = (s[0], s[1]-1)  
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.1
        nextState = (s[0], s[1]+1) 
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.1

    elif move == "v":
        nextState = (s[0]+1, s[1])
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.8
        nextState = (s[0], s[1]-1)  
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.1
        nextState = (s[0], s[1]+1) 
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.1

    elif move == "<":
        nextState = (s[0], s[1]-1)
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.8
        nextState = (s[0]-1, s[1])
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.1
        nextState = (s[0]+1, s[1])
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.1

    elif move == ">":
        nextState = (s[0], s[1]+1) 
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.8
        nextState = (s[0]-1, s[1]) 
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.1
        nextState = (s[0]+1, s[1])
        if not validmove(nextState):
            nextState = s
        if nextState == Sprime:
            temp += 0.1
    return temp

def validmove(state):
    if state[0] >= 0 and state[0] < len(environment_data):
        if state[1] >= 0 and state[1] < len(environment_data[1]):
            if environment_data[state[0]][state[1]] == 'X':
                return False
            return True
    return False

# MAIN

#load data
file = []
temp = csv.reader(open(sys.argv[1]))
for row in temp:
    file.append(row)

environment_data = file
non_terminal_reward = float(sys.argv[2])
gamma = float(sys.argv[3])
K = int(sys.argv[4])

#Value Iteration
N = len(environment_data), len(environment_data[0])
policy = np.chararray(N, unicode=True)
Uprime = np.zeros(N)
U = 0

for z in range(K):
    U = Uprime.copy()
    for i in range(N[0]):
        for j in range(N[1]):
            if environment_data[i][j] == 'X':
                Uprime[i][j] = 0
                policy[i][j] = 'x'
            elif environment_data[i][j] != '.':
                Uprime[i][j] = float(environment_data[i][j])
                policy[i][j] = 'o'
            else:
                Max = 0
                for move in directions:
                    value = 0
                    for k in range(N[0]):
                        for l in range(N[1]):
                            value += (transitions((k,l), (i,j), move) * U[k][l])
                    Max = max(value, Max)
                    if value == Max:
                        policy[i][j] = move
                Uprime[i][j] = R(i,j) + (gamma * Max)

#print out
print("utilities:")
for i in range(len(U)):
    printstring = ""
    for j in range(len(U[0])):
        printstring += "{:6.3f} ".format(U[i][j])
    print(printstring)

print("\npolicy:")
for i in range(len(policy)):
    printstring = ""
    for j in range(len(policy[0])):
        printstring += "{:6s}".format(policy[i][j])
    print(printstring)