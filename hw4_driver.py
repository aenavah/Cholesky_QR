import pandas as pd
import numpy as np
import copy
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

import LinAl
Cholesky = LinAl.Cholesky_Decomposition
Get_Data = LinAl.get_matrix
Get_Solution = LinAl.Substitution

if __name__== "__main__":
  data = "atkinson.dat"
  A = [[8.0, 3.22, .8, 0, 4.1],[3.22, 7.76 ,2.33 , 1.91, -1.03],[ .8, 2.33, 5.25, 1.0 ,3.02], [0.0 ,1.91 ,1.0 , 7.5 , 1.03], [4.1, -1.03, 3.02, 1.03, 6.44]]
  B = [[9.45, 0], [-12.20,0], [7.78,0], [-8.1, 0], [10.0, 0]]
  show_plot = 0
  perform_cholesky = 1
  #visualizing data
  if show_plot == 1:
    dd = np.loadtxt(data, skiprows = 0)
    plt.plot(dd[:,0], dd[:,1],'o-')
    plt.show()

  #Cholesky-----
  if perform_cholesky == 1:
    LU = Cholesky(A, " ", " ")
    #going to try numpy again i am brave
    A = np.array(A)
    B = np.array(B)
    for column in range(B.shape[0]):
      b = B[:, column] #columns of B
      print(b) 
      Get_Solution(A, b)
      break 
    
    