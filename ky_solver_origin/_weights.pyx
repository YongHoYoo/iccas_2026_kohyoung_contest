cdef extern from "_weights_data.h":
    const unsigned char w_open_data[]
    const unsigned int w_open_data_len
    const unsigned char w_closed_data[]
    const unsigned int w_closed_data_len

import zlib as _zlib
import io as _io
import numpy as _np

_CACHE = {}


def get_weights(model_type):
    if model_type not in _CACHE:
        if model_type == 'open':
            blob = bytes(w_open_data[:w_open_data_len])
        else:
            blob = bytes(w_closed_data[:w_closed_data_len])
        raw = _zlib.decompress(blob)
        _CACHE[model_type] = dict(_np.load(_io.BytesIO(raw)))
    return _CACHE[model_type]
