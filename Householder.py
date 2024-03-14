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
 
  R = np.triu(A)
  return Q, R

def backsub(U, b):

  m, n = np.shape(b)
  x = np.zeros_like(b)

  x[m-1, 0] = b[m-1, 0]/U[m-1, m-1]
  
  for i in range(m - 1, -1, -1):
    summ = 0.0
    for k in range(i+1, m):
      summ += U[i, k] * x[k, 0]
    x[i, 0] = (b[i, 0] - summ)/U[i, i]
  return x

def vandermonde(x, degree):
  num_samples = len(x)
  V = np.zeros((num_samples, degree + 1))

  for i in range(num_samples):
      for j in range(degree + 1):
        V[i, j] = x[i] ** j
  return V

def frobenius_norm(matrix):
  f = np.sqrt(np.sum(matrix**2))
  return f