# matrixdot_lut.py
import numpy as np
cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] _dot(np.ndarray[np.int_t, ndim=2] m1, np.ndarray[np.int_t, ndim=2] m2, np.ndarray[np.float32_t, ndim=2] lut,):
  cdef np.ndarray[np.float32_t, ndim=2] r
  cdef int i, j, k,
  cdef int m1_locate, m2_locate,
  cdef np.float32_t s
  if m1.shape[1] != m2.shape[0]:
    raise ValueError('m1 and m2 dimension mismatch')
  r = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
  for i in range(m1.shape[0]):
    for j in range(m2.shape[1]):
      s = 0
      for k in range(m1.shape[1]):
        m1_locate = m1[i,k]
        m2_locate = m2[k,j]
        s =  s + lut[m1_locate+128, m2_locate+128]
      r[i, j] = s
  return r


def dot(m1, m2, lut):
  return _dot(m1, m2, lut)