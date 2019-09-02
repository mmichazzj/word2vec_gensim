#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
from gensim.models import KeyedVectors
import logging
import multiprocessing
import os
import re
import sys

from nltk import word_tokenize
from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def cleanhtml(raw_html):
    # Remove html tag and content inside
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = os.path.join(root, filename)
                for line in open(file_path, 'r', encoding='utf-8'):
                    sline = line.strip()
                    if sline == "":
                        continue
                    rline = cleanhtml(sline)
                    tokenized_line = word_tokenize(rline)
                    if tokenized_line:
                        is_alpha_word_line = [word.lower() for word in tokenized_line]
                        yield is_alpha_word_line


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Please use python train_word2vec_with_gensim.py data_path")
        exit()
    data_path = sys.argv[1]
    begin = time()

    sentences = MySentences(data_path)
    model = gensim.models.Word2Vec(sentences,
                                   size=32,"embedding size
                                   window=5,
                                   min_count=10,
                                   workers=multiprocessing.cpu_count())

    # Create target Directory if don't exist
    dirName = 'data/model'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    # Save trained model and word embedding
    model.save("data/model/word2vec.model")
    model.wv.save("data/model/wordvectors.kv")

    end = time()
    print("Total procesing time: %d seconds"% (end - begin))
