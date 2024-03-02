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
  A = [[2.0, -1.0, 0.0], 
       [-1.0, 2.0, -1.0],
       [0.0, -1.0, 2.0]]
  B = [[2,1],
       [-4,-2],
       [3,5]]
  
  perform_cholesky_test = 0
  show_plot = 0
  polyfitting = 1

  'cholesky----------'
  if perform_cholesky_test == 0:
    A_original, X, singular, spd  = cholesky_solve(A,B)
  #print(A_original @ X) #returns B, -> sub works

  data = "atkinson.dat"
  dd = np.loadtxt(data, skiprows = 0)
  dd = np.array(dd)
  'visualizing data'
  if show_plot == 1:
    plt.plot(dd[:,0], dd[:,1],'o-')
    plt.show()

  if polyfitting == 1:
    #test = np.array([[0, 10],[1, -2],[2, 3],[3, -10],[4, 16]]) 
    #test = vandermonde(test[:, 0]) #-> works
    x_coords = dd[:, 0]
    y_coords = dd[:, 1]
    X_v = vandermonde(x_coords)
    print(X_v)
    Y_v = vandermonde(y_coords)
    print(Y_v)
    A_original, sol, singular, spd = cholesky_solve(X_v, Y_v)
    #print(sol)
    #print(np.linalg.norm(sol))
    #print(dd)

