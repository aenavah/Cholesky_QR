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
  #np.set_printoptions(precision = 2) #no ugly 
  data = "atkinson.dat"
  dd = np.loadtxt(data, skiprows = 0)
  data = np.array(dd).copy()

  '''Q1'''
  show_ogplot = 0 #to show plot before polynomial fitting
  polyfitting_plot = 0 #Cholesky off 


  if show_ogplot == 1:
    plt.plot(dd[:,0], dd[:,1],'o-')
    plt.show()

  if polyfitting_plot == 1:
    xs = data[:, 0].copy()
    b_row = data[:, 1].copy()
    b_copy = b_row.reshape(-1,1).copy()

    poly = 3
    V = vandermonde(xs, poly).copy()
    VT = np.transpose(V).copy()

    # V^T @ V
    VT_V = np.dot(VT,V)      # checked by calc -> correct
    VT_b = np.dot(VT,b_copy) # checked by calc -> correct

    VT_V_c = VT_V.copy()
    VT_b_c = VT_b.copy()
    # Cholesky Decomposition on V^T V 
    L_LT, singular, spd = cholesky_decomp(VT_V, "", "") #checked by calculator! 
    coefs, singular = chol_sub(L_LT, VT_b)

    coefs_c = coefs.copy()
    L_LT_c = L_LT.copy()
    # Coefficients and Plot
    print("Coefficients of the polynomial with degree " + str(poly) + ":")
    print(coefs)
    plt.plot(xs, np.dot(V,coefs), color = "pink")
    plt.scatter(xs, b_copy, color = "black", s = 3)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(str(poly) + " degree polynomial of Atkinson data")
    #plt.show()

    #norm of V^TV@x - V^T@y
    print("----------------------------------")

    '''Q2'''
  poly = 5

  xs = data[:, 0].copy()
  b = data[:, 1].reshape(-1,1).copy()

  V = vandermonde(xs, poly)
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

  #reduce Q and R:
  m_r,n_r = np.shape(R)
  for i in range(m_r):
    if sum(R[i,:]) == 0:
      index = i
      break 

  Q_copy = Q.copy()
  R_copy = R.copy()

  R_hat = R[:i, :]
  Q_hat = Q[:, :i] # is my Q not tranpose?

  # Solving Rhat@x = Qhat.T@b:

  # if "np_qr" == "np_qr": #comparing numpy reduced with my reduced 
    # Q_sp, R_sp = np.linalg.qr(V, mode = "reduced") 
    # print(frobenius(Q_hat - Q_sp))

  print("\nSolution x of Rhat@x = Qhat^T@b :")
  x_sol = backsub(R_hat, Q_hat.T @ b)
  x_sol_T = x_sol.reshape(1, -1)
  print(x_sol_T) 

  print("\n||Rhat@x - Qhat^T@b||:")
  print(frobenius(R_hat @ x_sol - Q_hat.T @ b))

  plt.plot(xs, np.dot(V,x_sol), color = "pink")
  plt.scatter(xs, b, color = "black", s = 3)
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.title("Degree " + str(poly) + " polynomial of Atkinson data with Householder Method")
  #plt.show()