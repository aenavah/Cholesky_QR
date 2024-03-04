import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy

#modules:
import Cholesky
cholesky_decomp = Cholesky.Cholesky_Decomposition
get_data = Cholesky.get_matrix = Cholesky.Substitution
chol_sub = Cholesky.Substitution
vandermonde = Cholesky.Vandermonde
frobenius_norm = Cholesky.frobenius_norm

import Householder
vandermonde = Householder.vandermonde
householder = Householder.householder
backsub = Householder.backsub
frobenius = Householder.frobenius_norm

# def cholesky_solve(A,B):
#   A = np.array(A)
#   A_original = np.copy(A) #wtf 
#   B = np.array(B)
#   LU, singular, spd = cholesky_decomp(A, " ", " ")
#   #going to try numpy again 
#   LU = np.array(LU)
#   #iterating through B columns in AX=B
#   X = []
#   print("Performing substitution...")
#   for column in range(B.shape[1]):
#     #print("Working on column " + str(column) + " of B...")
#     b = B[:, column] 
#     x, singular = chol_sub(LU, b)
#     X.append(x)
#   X = np.array(X)
#   X = np.transpose(X)
#   if singular == True:
#     spd = False
#   return A_original, X, singular, spd 

if __name__== "__main__":

  data = "atkinson.dat"
  dd = np.loadtxt(data, skiprows = 0)
  data = np.array(dd).copy()

  '''Q1'''
  show_ogplot = 0 #to show plot before polynomial fitting
  polyfitting_plot = 0 #show fitted plot


  if show_ogplot == 1:
    plt.plot(dd[:,0], dd[:,1],'o-')
    plt.show()

  if polyfitting_plot == 1:
    xs = data[:, 0].copy()
    b_row = data[:, 1].copy()
    b_copy = b_row.reshape(-1,1).copy()

    poly = 5
    V = vandermonde(xs, poly).copy()
    VT = np.transpose(V).copy()

    # V^T @ V
    VT_V = np.dot(VT,V)      # checked by calc -> correct
    VT_b = np.dot(VT,b_copy) # checked by calc -> correct

    # Cholesky Decomposition on V^T V 
    L_LT, singular, spd = cholesky_decomp(VT_V, "", "") #checked by calculator! 
    coefs, singular = chol_sub(L_LT, VT_b)

    # Coefficients and Plot
    print("Coefficients of the polynomial with degree " + str(poly) + ":")
    print(coefs)
    plt.plot(xs, np.dot(V,coefs), color = "pink")
    plt.scatter(xs, b_copy, color = "black", s = 3)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(str(poly) + " degree polynomial of Atkinson data")
    plt.show()

    '''Q2'''
  poly = 3

  xs = data[:, 0].copy()
  b = data[:, 1].reshape(-1,1).copy()
  V = vandermonde(xs, 3)
  Q, R = householder(V) #Q orthogonal, R upper triangular #Q IS TRANSPOSE ALREADY

  #A - QR
  print("A - QR:")
  QR_diff = V - Q@R
  print(QR_diff)

  #||A - QR||_F
  print("\n||A - QR||_F :")
  print(frobenius(QR_diff))

  #Q^TQ - I
  print("\nQ^TQ - I :")
  print((Q@Q.T) - np.eye(np.size(Q[0])))

  #||Q^TQ - I||
  print("\n||Q^TQ - I|| :")
  print(frobenius((Q@Q.T) - np.eye(np.size(Q[0]))))

  #Couldn't reduce it :(
  #reduce Q and R:
  m_r,n_r = np.shape(R)
  for i in range(m_r):
    if sum(R[i,:]) == 0:
      index = i
      break 
  #R_hat = R[0:index, :]
  #Q_hat = Q[:, 0:index].copy
  #Q[:, index:] = 0.0
