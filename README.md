# word2vec_gensim
Train model based on Wikipedia English corpus with gensim package.


### Wikipedia dump

Download the latest English wikipedia article corpus from [here](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). Its size is about 15G.


### Wikipedia dump extraction

The original wikipedia dump that can be downloaded is in xml format. Thus we need to use a extractor tool to parse it. The one I used is from the [wikiextractor](https://github.com/attardi/wikiextractor) repository. Only the file *WikiExtractor.py* is needed and the descriptions of parameters can be found the in the repository readme file. The output would be each article id and its name followed by the content in text format.

```
python WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -b 1G -o extracted --no-template --processes 24
```

### Text pre-processing and word2vec training

Before the word2vec training, the corpus needs to be pre-processed, which bascially includes: extracting raw text, word tokenization and lower case. For example, original document maybe like this:

```
<doc id="4792" url="https://en.wikipedia.org/wiki?curid=4792" title="Barry Goldwater">
Barry Goldwater

Barry Morris Goldwater (January 2, 1909 – May 29, 1998) was an American politician, businessman, and author who was a five-term Senator from Arizona (1953–1965, 1969–1987) and the Republican Party nominee for president of the United States in 1964. Despite his loss of the 1964 presidential election in a landslide, Goldwater is the politician most often credited with having sparked the resurgence of the American conservative political movement in the 1960s. He also had a substantial impact on the libertarian movement.
```
 
After pre-processing, we can get word tokens like this:

```
['barry', 'goldwater'], ['barry', 'morris', 'goldwater', '(', 'january', '2', ',', '1909', '–', 'may', '29', ',', '1998', ')', 'was', 'an', 'american', 'politician', ...]
```

You can start text pre-processing and training with gensim:

```
python train_word2vec_with_gensim.py extracted
```

### Load trained model and word embedding
Trained model base on sample data can be found under model folder, you can load the model like this:

```
model = gensim.models.Word2Vec.load("model/word2vec.model")
```

or directly load word embeddings:

```
wv = gensim.models.KeyedVectors.load("model/wordvectors.kv", mmap='r')
```
