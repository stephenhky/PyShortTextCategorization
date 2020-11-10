from setuptools import setup, Extension
import numpy as np


try:
    from Cython.Build import cythonize
    ext_modules = cythonize(['shorttext/metrics/dynprog/dldist.pyx',
                             'shorttext/metrics/dynprog/lcp.pyx',
                             'shorttext/spell/edits1_comb.pyx'])
except ImportError:
    ext_modules = [Extension('shorttext.metrics.dynprog.dldist', ['shorttext/metrics/dynprog/dldist.c']),
                   Extension('shorttext.metrics.dynprog.lcp', ['shorttext/metrics/dynprog/lcp.c']),
                   Extension('shorttext.spell.edits1_comb', ['shorttext/spell/edits1_comb.c'])]


def package_description():
    text = open('README.md', 'r').read()
    startpos = text.find('## Introduction')
    return text[startpos:]


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


def setup_requirements():
    return [package_string.strip() for package_string in open('setup_requirements.txt', 'r')]


def test_requirements():
    return [package_string.strip() for package_string in open('test_requirements.txt', 'r')]



setup(name='shorttext',
      version='1.4.3',
      description="Short Text Mining",
      long_description=package_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Text Processing :: Linguistic",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Cython",
          "Programming Language :: C",
          "Natural Language :: English",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research"
      ],
      keywords="shorttext natural language processing text mining",
      url="https://github.com/stephenhky/PyShortTextCategorization",
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      ext_modules=ext_modules,
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
                'shorttext.metrics.transformers',
                'shorttext.metrics.embedfuzzy',
                'shorttext.spell'],
      package_dir={'shorttext': 'shorttext'},
      package_data={'shorttext': ['data/*.csv', 'utils/*.txt',
                                  'metrics/dynprog/*.pyx', 'metrics/dynprog/*.c',
                                  'spell/*.pyx', 'spell/*.c']},
      include_dirs=[np.get_include()],
      setup_requires=setup_requirements(),
      install_requires=install_requirements(),
      scripts=['bin/ShortTextCategorizerConsole',
               'bin/ShortTextWordEmbedSimilarity',
               'bin/WordEmbedAPI'],
      test_suite="test",
      tests_requires=test_requirements(),
      zip_safe=False)
