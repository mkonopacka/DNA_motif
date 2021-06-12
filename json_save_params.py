import json 
import numpy as np
import argparse 
 
# position weight matrix:
tmp = np.array([
    [3/8,1/8,2/8,2/8],
    [1/10,2/10,3/10,4/10],
    [1/7,2/7,1/7,3/7]])

Theta = tmp.T

# background distribution
ThetaB = np.array([1/4,1/4,1/4,1/4])
params = {
    "w" : 3,
    "alpha" : 0.5,
    "k" : 10,
    "Theta" : Theta.tolist(),
    "ThetaB" : ThetaB.tolist()
    }

#Uwaga: powyzej nie mozna podac macierzy -- tutaj sa zamieniane na listy
#(potem po wczytaniu, wystarczy zamienic na macierz np.asarray(.))
with open('params_set1.json', 'w') as outfile:
    json.dump(params, outfile)

    