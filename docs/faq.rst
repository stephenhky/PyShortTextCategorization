Frequently Asked Questions (FAQ)
================================

**Q1. Can we use backends other than TensorFlow?**

Ans: No.


**Q2. Can we use word-embedding algorithms other than Word2Vec?**

Ans: Yes. Besides Word2Vec, you can use FastText and Poincaré embedding. See: :doc:`tutorial_wordembed` .


**Q3. Can this package work on Python 2?**

Ans: No.



**Q4. How should I cite `shorttext` if I use it in my research?**

Ans: For the time being, You do not have to cite a particular paper for using this package.
However, if you use any particular functions or class, check out the docstring. If there is a paper (or papers)
mentioned, cite those papers. For example, if you use `CNNWordEmbed` in `frameworks
<https://github.com/stephenhky/PyShortTextCategorization/blob/master/shorttext/classifiers/embed/nnlib/frameworks.py>`_,
according to the docstring, cite Yoon Kim's paper. Refer to this documentation for the reference too.


**Q5. I am having trouble in install `shorttext` on Google Cloud Platform. What should I do?**

Ans: There is no "Python.h". Run: `sudo apt-get install python3-dev` in SSH shell of the VM instance.



**Q6. Where is the Sakiguchi spell corrector? **

Ans: It was removed since release 3.0.0, but you can refer to the `examples\` folder in the
Github repository for the codes.



**Q7. Where are `WrappedBERTEncoder` and `BERTScorer`?**

Ans: It was removed since release 3.0.0, but you can install another package `shorttext-bert`
to get the same functionality.



**Q8. My model files were created by `shorttext` version < 2.0.0. How do I make them readable for version >= 2.0.0?

Ans: Simply make those files with names ending with `.h5` to `.weights.h5`.



Home: :doc:`index`
