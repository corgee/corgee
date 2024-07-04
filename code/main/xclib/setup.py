from Cython.Build import cythonize
import numpy as np
from distutils.core import setup


setup(name='Sparse matrix operations app',
      ext_modules=cythonize("_sparse.pyx"), include_dirs=[np.get_include()])
