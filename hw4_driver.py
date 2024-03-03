import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#module:
import Cholesky
cholesky_decomp = Cholesky.Cholesky_Decomposition
get_data = Cholesky.get_matrix = Cholesky.Substitution
chol_sub = Cholesky.Substitution
vandermonde = Cholesky.Vandermonde
frobenius_norm = Cholesky.frobenius_norm

def cholesky_solve(A,B):
  A = np.array(A)
  A_original = np.copy(A) #wtf 
  B = np.array(B)
  LU, singular, spd = cholesky_decomp(A, " ", " ")
  #going to try numpy again 
  LU = np.array(LU)
  #iterating through B columns in AX=B
  X = []
  print("Performing substitution...")
  for column in range(B.shape[1]):
    #print("Working on column " + str(column) + " of B...")
    b = B[:, column] 
    x, singular = chol_sub(LU, b)
    X.append(x)
  X = np.array(X)
  X = np.transpose(X)
  if singular == True:
    spd = False
  return A_original, X, singular, spd 

if __name__== "__main__":
  Test_A = [
       [2.0, -1.0, 0.0], 
       [-1.0, 2.0, -1.0],
       [0.0, -1.0, 2.0]]
  Test_B = [
       [2,1],
       [-4,-2],
       [3,5]]
  
  perform_cholesky_test = 0
  show_ogplot = 1
  polyfitting_plot = 1

  'cholesky----------'
  if perform_cholesky_test == 0:
    A0_test, X_test, singular, spd  = cholesky_solve(Test_A,Test_B)
  # print(A_original @ X) #returns B, -> sub works

  data = "atkinson.dat"
  dd = np.loadtxt(data, skiprows = 0)
  data = np.array(dd).copy()
  'visualizing data'

  if show_ogplot == 0:
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
    plt.title(str(poly) + "th degree polynomial of Atkinson data")
    plt.show()