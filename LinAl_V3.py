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
  print(A)

def Cholesky_backsub(A, dim, B):
  m_b = len(B)
  n_b = len(B[0])
  m = len(A)
  n = len(A[0])
  X = [] #size B 
  for tmp_0 in range(0, n_b):
    tmp_list = []
    for tmp_1 in range(0,m_b):
      tmp_list.append("")
    X.append(tmp_list)

  for column in range(0, n_b):
    x = []
    y = []
    b = []
    for row in range(0, m_b):
      b.append(B[row][column])
      x.append(0)
      y.append(0)
      #here b and x are column vectors
      for i in range(0,m):
        sum = b[i]
        for j in range(0,i-1):
          sum -= y[j]*A[i][j]
  print("b", b)
    
if __name__ == "__main__":
  #visualizing data
  data = "atkinson.dat"
  dd = np.loadtxt(data, skiprows = 0)
  plt.plot(dd[:,0], dd[:,1],'o-')
  #plt.show()

  #Cholesky
  A = get_matrix(data).tolist() #we dont like numpy anymore, list of list 
  Test = [[6,15,25],[15,55,225],[55,225,979]]
  L = Cholesky_Decomp(Test, " ") #Good!
  #Q: check if singular? 
  Test2 = [[6,15,25],[15,55,225]]
  Cholesky_backsub(L, 2, Test2)

