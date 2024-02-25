import pandas as pd
import numpy as np
import copy
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

#turn matrix into python readable
def get_matrix(myFileName):
  with open(myFileName, 'r') as myFile:
    data = myFile.read()
    matrix = np.loadtxt(myFileName)
    return matrix


def Cholesky_Decomp(A, problem): 
  problem == False
  n = len(A)#rows
  m = len(A[0])#columns
  for j in range(0,m): #from 0 to m-1
    for k in range(0,j):
      A[j][j] -= A[j][k]**2
    if A[j][j]>=0:
      A[j][j] = A[j][j]**(1/2)
    else:
      problem == True
    for i in range(j+1, m):
      for k_2 in range(0,j):
        A[i][j] -= A[i][k_2]*A[j][k_2]
      A[i][j]/=A[j][j]
  #print(A)
  return A

def Cholesky_backsub(L, dim, B):
  m_b, n_b = np.array(B).shape
  m, n = np.array(L).shape
  X = np.zeros_like(B).tolist() #size B
  Y = np.zeros_like(B).tolist() 
  #print(X)

  for column in range(0, n_b):#iterating through vectors of B 
    #forward sub 
    for i in range(0,m):
      summ = B[i][column]
      for j in range(0,i):
        summ -= Y[j][column]*L[i][j]  
      Y[i][column] = summ/L[i][i]
    #backsub
    for i in range(m-1, -1, -1):
      print("i", i+1)
      for k in range(i+1, m):
        print("k", k+1)
        Y[i][column] = Y[i][column] - np.conjugate(L[k][i])*X[k][column]
      X[i][column] = Y[i][column]/np.conjugate(L[i][i])
    return Y, X


 
if __name__ == "__main__":
  #visualizing data
  data = "atkinson.dat"
  dd = np.loadtxt(data, skiprows = 0)
  plt.plot(dd[:,0], dd[:,1],'o-')
  #plt.show()

  #Cholesky
  A = get_matrix(data).tolist() #we dont like numpy anymore, list of list 
  #decomp 
  Test = [[6,15,25],[15,55,225],[55,225,979]]
  L = Cholesky_Decomp(Test, " ") #Good!
    #Q: check if singular?
  #backsub 
  Test2 = [[1,2,3],[5,5,6],[8,10,9]]
  Y, X = Cholesky_backsub(L, 2, Test2)
  print("X--", X)
  print((Y[1][0] - L[1][0]*(Y[0][0]/L[0][0]))/L[1][1]) #should be x10 but its NOT
  #backsub works fine, X[0][0] is found fine, after is sad
  # testing backsub by multplying random Y and L and checking if = B print(L[2][0]*Y[0][1] + L[2][1]*Y[1][1] + L[2][2]*Y[2][1])