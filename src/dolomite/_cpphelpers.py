# DO NOT MODIFY: this is automatically generated by the cpptypes

import os
import ctypes as ct
import numpy as np

def _catch_errors(f):
    def wrapper(*args):
        errcode = ct.c_int32(0)
        errmsg = ct.c_char_p(0)
        output = f(*args, ct.byref(errcode), ct.byref(errmsg))
        if errcode.value != 0:
            msg = errmsg.value.decode('ascii')
            lib.free_error_message(errmsg)
            raise RuntimeError(msg)
        return output
    return wrapper

# TODO: surely there's a better way than whatever this is.
dirname = os.path.dirname(os.path.abspath(__file__))
contents = os.listdir(dirname)
lib = None
for x in contents:
    if x.startswith('_core') and not x.endswith("py"):
        lib = ct.CDLL(os.path.join(dirname, x))
        break

if lib is None:
    raise ImportError("failed to find the _core.* module")

lib.free_error_message.argtypes = [ ct.POINTER(ct.c_char_p) ]

def _np2ct(x, expected, contiguous=True):
    if not isinstance(x, np.ndarray):
        raise ValueError('expected a NumPy array')
    if x.dtype != expected:
        raise ValueError('expected a NumPy array of type ' + str(expected) + ', got ' + str(x.dtype))
    if contiguous:
        if not x.flags.c_contiguous and not x.flags.f_contiguous:
            raise ValueError('only contiguous NumPy arrays are supported')
    return x.ctypes.data

lib.py_fetch_booleans.restype = ct.c_uint8
lib.py_fetch_booleans.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_uint8,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_numbers.restype = ct.c_uint8
lib.py_fetch_numbers.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_strings.restype = ct.c_uint8
lib.py_fetch_strings.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_char_p,
    ct.c_uint8,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_csv.restype = None
lib.py_free_csv.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_column_stats.restype = None
lib.py_get_column_stats.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_string_stats.restype = ct.c_uint8
lib.py_get_string_stats.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_load_csv.restype = ct.c_void_p
lib.py_load_csv.argtypes = [
    ct.c_char_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

def fetch_booleans(ptr, column, contents, pop):
    return _catch_errors(lib.py_fetch_booleans)(ptr, column, _np2ct(contents, np.uint8), pop)

def fetch_numbers(ptr, column, contents, mask, pop):
    return _catch_errors(lib.py_fetch_numbers)(ptr, column, _np2ct(contents, np.float64), _np2ct(mask, np.uint8), pop)

def fetch_strings(ptr, column, contents, pop):
    return _catch_errors(lib.py_fetch_strings)(ptr, column, contents, pop)

def free_csv(ptr):
    return _catch_errors(lib.py_free_csv)(ptr)

def get_column_stats(ptr, column, type, size, loaded):
    return _catch_errors(lib.py_get_column_stats)(ptr, column, type, size, loaded)

def get_string_stats(ptr, column, lengths, mask):
    return _catch_errors(lib.py_get_string_stats)(ptr, column, _np2ct(lengths, np.int32), _np2ct(mask, np.uint8))

def load_csv(path):
    return _catch_errors(lib.py_load_csv)(path)