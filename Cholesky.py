import pandas as pd
import numpy as np
import copy
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

global emach 
emach = 10**-16

'''put matrix into python form'''
def get_matrix(myFileName):
  with open(myFileName, 'r') as myFile:
    data = myFile.read()
    matrix = np.loadtxt(myFileName)
    return matrix
  
'''Input: square A, singular: binary, spd by using decomposition'''
def Cholesky_Decomposition(A, singular, spd):
  if type(A) == np.ndarray:
    A.tolist()
  spd = True
  singular == False
  n = len(A)#rows
  m = len(A[0])#columns
  for j in range(0,m): #from 0 to m-1
    for k in range(0,j):
      A[j][j] -= A[j][k]**2
    if A[j][j] >= emach:
      A[j][j] = A[j][j]**(1/2)
    else:
      singular = True
      spd = False
    for i in range(j+1, m):
      for k_2 in range(0,j):
        A[i][j] -= A[i][k_2]*A[j][k_2]
      A[i][j]/=A[j][j]
  return A, singular, spd 

'''
cholesky substitution: 
Forward substitution solves for y st Ly = b 
backward substitution solves for L^{*}x=y
'''
def Substitution(LU, b):
  print()
  b_copy = b.copy()
  singular = False
  L = np.tril(LU).copy()
  Lstar = np.transpose(L).copy()
  #print(L@Lstar) #returns A, -> decomp works 
  m,n = np.shape(LU)
  x = np.zeros(m)
  y = np.zeros(m)

  ### Forward Substitution 
  for i in range(0, m):
    summ = b[i]
    for j in range(i):
      summ -= y[j]*L[i,j]
    y[i] = summ/L[i,i]
  #print(L@y) #returns b -> forward sub works 
  #print(y) # -> to confirm backsub works

  ### Backward Substitution
  for i in range(m - 1, -1, -1):
    if L[i,i] <= emach:
      singular = True
    if Lstar[i,i] <= emach:
      singular = True
    for k in range(i + 1, m):
      y[i] -= Lstar[i, k] * x[k] 
    x[i] = y[i] / Lstar[i,i]
  #print(Lstar @ x) #returns y -> backsub works
  return x, singular

def Vandermonde(x, degree):
    num_samples = len(x)
    V = np.zeros((num_samples, degree + 1))

    for i in range(num_samples):
        for j in range(degree + 1):
            V[i, j] = x[i] ** j
    return V

def frobenius_norm(matrix):
  f = np.sqrt(np.sum(matrix**2))
  return f