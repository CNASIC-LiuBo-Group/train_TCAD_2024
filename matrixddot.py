# matrixdot.py
import numpy as np
hat = 10000
wat = 25
hbt = 25
wbt = 6
at = np.random.randint(low=0,high=256,size=(hat,wat))
bt = np.random.randint(low=0,high=256,size=(hbt,wbt))
def dot(m1, m2):
  if m1.shape[1] != m2.shape[0]:
    raise ValueError('m1 and m2 dimension mismatch')
  r = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
  for i in range(m1.shape[0]):
    for j in range(m2.shape[1]):
      s = 0
      for k in range(m1.shape[1]):
        s += m1[i, k] * m2[k, j]
      r[i, j] = s
  return r

dot(at,bt)