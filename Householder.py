import numpy as np 
import scipy

def householder(A):
  m,n = np.shape(A)
  
  for j in range(0, n):
    vj = np.zeros(m)
    
    sign_sj = np.sign(A[j,j])
    norm = np.sqrt(sum(A[j:,j]**2)) #from j on 
    s_j = sign_sj * norm 

    vj[j] = A[j, j] + s_j

    for i in range(j + 1, m):
      vj[i] = A[i,j]

    vj = np.transpose(vj)
    vj_norm = np.sqrt(sum(vj[:])**2)
    
    print("vj_norm", vj_norm)
    vj = vj/vj_norm
    vj_T = np.transpose(vj)


    A = A - 2*(vj @ vj_T) * A
  return A


test = np.array([
  [1, 2],
  [3, 4],
  [5, 6]])

H = householder(test)
print(H)
Q, R = scipy.linalg.qr(test)
print(R)