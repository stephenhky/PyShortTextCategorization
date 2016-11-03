from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='shorttext',
      version="0.0",
      description="Short Text Categorization using Embedded Word Vectors",
      long_description="Supervised learning algorithms for short text categorization using embedded word vectors such as Word2Vec",
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
      packages=['shorttext'],
      install_requires=[
          'numpy', 'scipy', 'keras', 'theano', 'nltk', 'gensim', 'pandas',
      ],
      scripts=['shorttext/ShortTextCategorizer.py',
               'shorttext/ShortTextCategorizerConsole.py',
               'shorttext/ShortTextCategorizerModelTraining.py'],
      include_package_data=True,
      zip_safe=False)