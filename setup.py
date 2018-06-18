from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    ext_modiles = cythonize(['shorttext/metrics/dynprog/dldist.pyx',
                             'shorttext/metrics/dynprog/lcp.pyx'])
except ImportError:
    ext_modiles = [Extension('_dldist', 'shorttext/metrics/dynprog/dldist.c'),
                   Extension('_lcp', 'shorttext/metrics/dynprog/lcp.c')]

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='shorttext',
      version="0.7.2",
      description="Short Text Mining",
      long_description="Supervised learning algorithms for short text categorization using embedded word vectors such as Word2Vec, or immediate feature vectors using topic models",
      classifiers=[
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Natural Language :: English",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Programming Language :: Python :: 2.7",
          "License :: OSI Approved :: MIT License",
      ],
      keywords="short text natural language processing text mining",
      url="https://github.com/stephenhky/PyShortTextCategorization",
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      ext_modules=ext_modiles,
      packages=['shorttext',
                'shorttext.utils',
                'shorttext.classifiers',
                'shorttext.classifiers.embed',
                'shorttext.classifiers.embed.nnlib',
                'shorttext.classifiers.embed.sumvec',
                'shorttext.classifiers.bow',
                'shorttext.classifiers.bow.topic',
                'shorttext.classifiers.bow.maxent',
                'shorttext.data',
                'shorttext.stack',
                'shorttext.generators',
                'shorttext.generators.bow',
                'shorttext.generators.charbase',
                'shorttext.generators.seq2seq',
                'shorttext.metrics',
                'shorttext.metrics.dynprog',
                'shorttext.metrics.wasserstein',
                'shorttext.metrics.embedfuzzy',
                'shorttext.spell'],
      package_dir={'shorttext': 'shorttext'},
      package_data={'shorttext': ['data/*.csv', 'utils/*.pkl', 'metrics/dynprog/*.pyx']},
      include_dirs=[np.get_include()],
      setup_requires=['numpy>=1.11.3', 'scipy>=0.18.1'],
      install_requires=[
          'Cython', 'numpy>=1.11.3', 'scipy>=0.18.1', 'scikit-learn', 'keras>=2.0.0', 'gensim>=3.2.0',
          'pandas', 'spacy>=1.7.0', 'stemming', 'pulp',
      ],
      tests_require=[
          'unittest2', 'keras>=2.0.0', 'gensim>=3.2.0', 'tensorflow',
      ],
      scripts=['bin/ShortTextCategorizerConsole',
               'bin/ShortTextWordEmbedSimilarity',
               'bin/switch_kerasbackend'],
      # include_package_data=False,
      test_suite="test",
      zip_safe=False)
