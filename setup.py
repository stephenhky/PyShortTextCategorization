from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='shorttext',
      version="0.2.0",
      description="Short Text Categorization",
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
      packages=['shorttext',
                'shorttext.utils',
                'shorttext.classifiers',
                'shorttext.classifiers.embed',
                'shorttext.classifiers.embed.nnlib',
                'shorttext.classifiers.embed.sumvec',
                'shorttext.classifiers.bow',
                'shorttext.classifiers.bow.topic',
                'shorttext.data',],
      package_dir={'shorttext': 'shorttext'},
      package_data={'shorttext': ['data/*.csv']},
      setup_requires=['numpy', 'theano'],
      install_requires=[
          'numpy', 'scipy', 'scikit-learn', 'theano', 'keras', 'nltk', 'gensim', 'pandas',
      ],
      scripts=['bin/ShortTextDNNEvaluator',
               'bin/ShortTextDNNCategorizerConsole'],
      # include_package_data=False,
      zip_safe=False)
