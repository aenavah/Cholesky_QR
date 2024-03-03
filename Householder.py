import numpy as np 
import scipy

import numpy as np
import scipy.linalg

def householder(A):
    m, n = np.shape(A)
    
    Q = np.eye(m)
    for j in range(0, n):
      vj = np.zeros(m)
      
      sign_sj = np.sign(A[j, j])
      norm = np.sqrt(sum(A[j:, j]**2))  #[j,m)
      s_j = sign_sj * norm

      vj[j] = A[j, j] + s_j

      for i in range(j + 1, m):
        vj[i] = A[i, j]

      vj = vj.reshape((-1, 1))
      vj_norm = np.sqrt((np.sum(vj**2)))
      
      vj = vj / vj_norm
      vj_T = np.transpose(vj)

      A = A - 2 * (vj @ vj_T @ A)
      Q = Q @ (np.eye(m) - 2 *(vj @ vj_T))
    R = A
    return Q, R

test = np.array([
    [1, 2],
    [3, 4],
    [5, 6]])

Q_mine, R_mine = householder(test)
#Q, R = scipy.linalg.qr(test)
#print(Q_mine @ R_mine) # = A -> right 

