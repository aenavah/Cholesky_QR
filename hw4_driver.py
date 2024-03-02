import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#module:
import LinAl
cholesky_decomp = LinAl.Cholesky_Decomposition
get_data = LinAl.get_matrix = LinAl.Substitution
chol_sub = LinAl.Substitution

if __name__== "__main__":
  data = "atkinson.dat"
  A = [[2.0, -1.0, 0.0], 
       [-1.0, 2.0, -1.0],
       [0.0, -1.0, 2.0]]
  B = [[2,0],
       [-4,0],
       [3,0]]
  show_plot = 0
  perform_cholesky = 1

  'visualizing data'
  if show_plot == 1:
    dd = np.loadtxt(data, skiprows = 0)
    plt.plot(dd[:,0], dd[:,1],'o-')
    plt.show()

  'cholesky----------'
  if perform_cholesky == 1:
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
  #print(A_original@X) #returns B, -> sub works

