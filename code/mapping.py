#!/usr/bin/python3

##################
## Syntax
##################
## d will mean the dimensionality of the original feature space
## D will mean the dimensionality of the new feature space

import numpy as np
# import math

# Generates de matrix W, which has D columns and d rows
# x11 x12 x13 ... x1D
# x21 x22 x23 ... x2D
# x31 x32 x33 ... x3D
# .
# .
# .
# xd1 xd2 xd3 ... xdD

def genW(variance, d, D):
    mat = [[np.random.normal(0, variance) for i in range(D)] for i in range(d)]
    return np.matrix(mat)



# Recieves the original data matrix and the number of desired columns of the
# newly created matrix, and returns one random mapping of the matrix

def RFFmapping(origMat, ncol):
    variance = origMat.var()
    d = len(origMat)
    D = ncol
    w = genW(variance, d, D)
    prod =  origMat * w
    sinMat = np.sin(prod)
    cosMat = np.cos(prod)
    join = np.hstack((cosMat,sinMat))
    scale = np.sqrt(1/D)
    retMat = scale * join
    return retMat



print("Program Running")

# Just for testing
m = np.matrix("1 2 3; 4 5 6; 7,8,9")
