import pandas as pd
import numpy as np
import copy
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


'''Input: square A, singular: binary, spd by using decomposition'''
def Cholesky_Decomposition(A, singular, spd):
  singular == False
  n = len(A)#rows
  m = len(A[0])#columns
  for j in range(0,m): #from 0 to m-1
    for k in range(0,j):
      A[j][j] -= A[j][k]**2
    if A[j][j]>=0:
      A[j][j] = A[j][j]**(1/2)
    else:
      singular == True
    for i in range(j+1, m):
      for k_2 in range(0,j):
        A[i][j] -= A[i][k_2]*A[j][k_2]
      A[i][j]/=A[j][j]
  #print(A)
  return A

if __name__ == "__main__":
  print("plsplsplspls")