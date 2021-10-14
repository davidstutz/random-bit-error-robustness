"""
C interface, through CFFI, for bit manipulations.
"""
import os
import cffi
import common.paths

ffi = cffi.FFI()
debug = False
use_openmp = True

BASE_CODE = os.path.dirname(os.path.abspath(__file__))
with open('%s/cffi.h' % BASE_CODE) as my_header:
    ffi.cdef(my_header.read())

with open('%s/cffi.c' % BASE_CODE) as my_source:
    if debug:
        ffi.set_source(
            '_cffi',
            my_source.read(),
            extra_compile_args=[ '-pedantic', '-Wall', '-g', '-O0'],
        )
    # -ffast-math assumes there are no nans or infs!
    # -O3 includes -ffast-math!
    # https://stackoverflow.com/questions/22931147/stdisinf-does-not-work-with-ffast-math-how-to-check-for-infinity
    else:
        if use_openmp:
            ffi.set_source(
                '_cffi',
                my_source.read(),
                extra_compile_args=['-fopenmp', '-D use_openmp', '-O3','-march=native'],
                extra_link_args=['-fopenmp'],
            )
        else:
            ffi.set_source('_cffi',
                my_source.read(),
                extra_compile_args=['-O3','-march=native'],
)

ffi.compile()
#ffi.compile(verbose=True)
from _cffi import *