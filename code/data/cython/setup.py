from Cython.Distutils import build_ext
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
import os


os.environ["CC"] = "g++"
ext_modules = [
    Extension("_read",
              ["_read.pyx", "_read_utils.cpp"],
              language='c++11',
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-lpthread', '-fopenmp'],
              libraries=["cnpy"],
              extra_objects=["build/libcnpy.so"]
              )
]
setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules, include_dirs=[np.get_include(), 'build'], )
