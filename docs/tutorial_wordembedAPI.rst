Word Embedding Models in API
============================

A lot of embedding models take a few minutes to load, and it would be desirable
for such a model to be loaded in the memory first. It is why such an API
has been developed.

Model Preloading
----------------

To preload the model, use the script `WordEmbedAPI` provided. In
the command-line shell / Terminal, type:

```
> WordEmbedAPI /path/to/GoogleNews-vectors-negative300.bin.gz
```

After a few minutes, it will be loaded.

For details about using `WordEmbedAPI`, please refer to: :doc:`scripts` .

Class for Preloaded Model
-------------------------

After the model is loaded, it can be used like other word-embedding models
using `RESTfulKeyedVectors`:

```
>>> import shorttext
>>> wmodel = shorttext.utils.wordembed.RESTfulKeyedVectors('http://localhost', port='5000')
```

This model can be used like other `gensim` `KeyedVectors`.



Home: :doc:`index`

