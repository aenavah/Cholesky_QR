import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#module:
import Cholesky
cholesky_decomp = Cholesky.Cholesky_Decomposition
get_data = Cholesky.get_matrix = Cholesky.Substitution
chol_sub = Cholesky.Substitution

def cholesky_solve(A,B):
  A = np.array(A)
  A_original = np.copy(A) #wtf 
  B = np.array(B)
  LU = cholesky_decomp(A, " ", " ")
  #going to try numpy again 
  LU = np.array(LU)
  #iterating through B columns in AX=B
  X = []
  for column in range(B.shape[1]):
    print("Working on column " + str(column) + " of B...")
    b = B[:, column] 
    x = chol_sub(LU, b)
    X.append(x)
  X = np.array(X)
  X = np.transpose(X)
  return A_original, X

if __name__== "__main__":
  data = "atkinson.dat"
  A = [[2.0, -1.0, 0.0], 
       [-1.0, 2.0, -1.0],
       [0.0, -1.0, 2.0]]
  B = [[2,1],
       [-4,-2],
       [3,5]]
  show_plot = 0
  perform_cholesky = 1

  'visualizing data'
  if show_plot == 1:
    dd = np.loadtxt(data, skiprows = 0)
    plt.plot(dd[:,0], dd[:,1],'o-')
    plt.show()

  'cholesky----------'
  if perform_cholesky == 1:
    A_original, X = cholesky_solve(A,B)
  print(A_original @ X) #returns B, -> sub works

