from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='shorttext',
      version="0.3.8",
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
                # 'shorttext.classifiers.bow.maxent',
                'shorttext.data',
                'shorttext.stack',
                'shorttext.generators',
                'shorttext.generators.bow'],
      package_dir={'shorttext': 'shorttext'},
      package_data={'shorttext': ['data/*.csv', 'utils/*.pkl']},
      setup_requires=['numpy'],
      install_requires=[
          'numpy', 'scipy', 'scikit-learn', 'keras', 'gensim', 'pandas', 'spacy', 'stemming',
      ],
      scripts=['bin/ShortTextCategorizerConsole',
               'bin/ShortTextWord2VecSimilarity',
               'bin/switch_kerasbackend'],
      # include_package_data=False,
      zip_safe=False)
