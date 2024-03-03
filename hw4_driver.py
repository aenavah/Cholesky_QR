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
#overdetermined = Cholesky.overdetermined

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
  Test_A = [[2.0, -1.0, 0.0], 
       [-1.0, 2.0, -1.0],
       [0.0, -1.0, 2.0]]
  Test_B = [[2,1],
       [-4,-2],
       [3,5]]
  
  perform_cholesky_test = 0
  show_ogplot = 1
  polyfitting_plot = 1

  'cholesky----------'
  if perform_cholesky_test == 0:
    A0_test, X_test, singular, spd  = cholesky_solve(Test_A,Test_B)
  #print(A_original @ X) #returns B, -> sub works

  data = "atkinson.dat"
  dd = np.loadtxt(data, skiprows = 0)
  data = np.array(dd).copy()
  'visualizing data'


  if show_ogplot == 0:
    plt.plot(dd[:,0], dd[:,1],'o-')
    plt.show()

  if polyfitting_plot == 1:
    test = np.array([[0.0, 10.0],
                     [1.0, -2.0],
                     [2.0, 3.0],
                     [3.0, -10.0],
                     [4.0, 16.0]]) 
    #test = vandermonde(test[:, 0]) # -> works https://developer.apple.com/documentation/accelerate/finding_an_interpolating_polynomial_using_the_vandermonde_method

    #data = test #testing
    xs = data[:, 0].copy()
    print("xs from data:", xs)
    b_row = data[:, 1].copy()
    b_copy = b_row.reshape(-1,1).copy()
    print("b from data:", b_copy)

    V = vandermonde(xs, 5).copy()
    #V = np.vander(xs, 5)
    print("Vandermonde of x:", V)
    VT = np.transpose(V).copy()
    print("Vandermonde transpose of x:", VT)

    #V^T @ V
    VT_V = np.dot(VT,V) #checked by calc -> correct
    VT_b = np.dot(VT,b_copy) #checked by calc -> correct

    #cholesky decomposition on V^T V 
    L_LT, singular, spd = cholesky_decomp(VT_V, "", "") #checked by calculator! 
    coefs, singular = chol_sub(L_LT, VT_b)
    print(VT_V)
    print(len(xs))
    plt.plot(xs, np.dot(V,coefs), "o-")
    plt.show()

