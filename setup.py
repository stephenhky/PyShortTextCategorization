
from setuptools import setup
import numpy as np


setup(
      include_dirs=[np.get_include()],
      scripts=['bin/ShortTextCategorizerConsole',
               'bin/ShortTextWordEmbedSimilarity']
      )
