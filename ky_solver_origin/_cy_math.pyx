import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrtf


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_layer_norm(np.ndarray[np.float32_t, ndim=2] x,
                   np.ndarray[np.float32_t, ndim=1] w,
                   np.ndarray[np.float32_t, ndim=1] b):
    cdef int d0 = x.shape[0], d1 = x.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] out = np.empty((d0, d1), dtype=np.float32)
    cdef int i, j
    cdef float mean_v, var_v, inv_std, diff
    cdef float inv_d1 = 1.0 / d1
    for i in range(d0):
        mean_v = 0.0
        for j in range(d1):
            mean_v += x[i, j]
        mean_v *= inv_d1
        var_v = 0.0
        for j in range(d1):
            diff = x[i, j] - mean_v
            var_v += diff * diff
        var_v *= inv_d1
        inv_std = 1.0 / sqrtf(var_v + 1e-5)
        for j in range(d1):
            out[i, j] = (x[i, j] - mean_v) * inv_std * w[j] + b[j]
    return out
