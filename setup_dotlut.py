# setup_dotlut.py
import numpy
from Cython.Build import cythonize
from distutils.core import setup, Extension

setup(
  ext_modules=cythonize(Extension('matrixdot_lut_cython',
                  sources=['matrixdot_lut.pyx'],
                  language='c',
                  include_dirs=[numpy.get_include()],
                  library_dirs=[],
                  libraries=[],
                  extra_compile_args=[],
                  extra_link_args=[]))
)