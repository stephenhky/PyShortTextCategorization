
from setuptools import setup
import numpy as np
from Cython.Build import cythonize

ext_modules = cythonize(['shorttext/metrics/dynprog/dldist.pyx'])


setup(
      ext_modules=ext_modules,
      include_dirs=[np.get_include()],
      scripts=['bin/ShortTextCategorizerConsole',
               'bin/ShortTextWordEmbedSimilarity']
      )
