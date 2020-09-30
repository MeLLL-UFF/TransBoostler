from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from ekphrasis.classes.segmenter import Segmenter
from collections import OrderedDict
from scipy import spatial
import numpy as np
import operator
import logging
import sys, os

import logging
logging.basicConfig(level=logging.INFO)

#segmenter using the word statistics from Wikipedia
seg = Segmenter(corpus="english")

class Transfer:

  def __init__(self):
    pass

  def build_model_array(self, data, model, relation='AVG'):
	  """Turn relations into a single array"""
    dict = {}
    for example in data:
      temp = []

      # Tokenize relation words
      predicate = seg.segment(example)
      for word in predicate.split():
        temp.append(model.wv[word.lower().strip()])

      if(relation == 'AVG'):
        predicate = get_arrays_avg(temp)
      elif(relation == 'MAX'):
        predicate = max(temp, key=operator.methodcaller('tolist'))
      elif(relation == 'MIN'):
        predicate = min(temp, key=operator.methodcaller('tolist'))
      elif(relation == 'CONCATENATE'):
        predicate = min(temp, key=operator.methodcaller('tolist'))
        maximum = max(temp, key=operator.methodcaller('tolist'))
        predicate = np.append(predicate, maximum)

      dict[example.rstrip()] = predicate
    return dict

  def get_cosine_similarities(self, source, target):
    """Calculate cosine similarity of embedded arrays"""

    similarity = {}
    for s in source:
      for t in target:
        key = s + ',' + t
        similarity[key] = 1 - spatial.distance.cosine(source[s], target[t])

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df

  def similarity_word2vec(self, source, target, model_path, method):
    """Embed relations using pre-trained word2vec"""

    # Load Google's pre-trained Word2Vec model.
    word2vecModel = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')

    source = self.build_model_array(source, word2vecModel, method=method)
    target = self.build_model_array(target, word2vecModel, method=method)

    return self.get_cosine_similarities(source, target)







