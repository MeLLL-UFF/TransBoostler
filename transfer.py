from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from ekphrasis.classes.segmenter import Segmenter
from collections import OrderedDict
from scipy import spatial
import utils as utils
import pandas as pd
import numpy as np
import operator
import logging
import sys
import os

import logging
logging.basicConfig(level=logging.INFO)

# segmenter using the word statistics from Wikipedia
seg = Segmenter(corpus="english")

class Transfer:

  def __init__(self):
    pass

  def get_arrays_avg(self, data):
    """
        Calculate the average fo word embeddings element-wise

        Args:
            data(array): an array containing word embeddings
       Returns:
            a single array generated by taking the element-wise average
    """

    N = len(data)
    avg = np.array(data[0])
    for i in range(1, N):
        avg += np.array(data[i])
    return np.divide(avg, N)

  def build_model_array(self, data, model, method='AVG'):
    """
        Turn relations into a single array

        Args:
            data(array): an array containing word embeddings
            model(object): embedding model
            method(str): method to compact arrays of embedded words
       Returns:
            a dictionary that the keys are the words and the values are single arrays of embeddings
    """

    dict = {}
    for example in data:
      temp = []

      # Tokenize words of relation
      predicate = seg.segment(example[0])
      for word in predicate.split():
        temp.append(model.wv[word.lower().strip()])
    
      if(method == 'AVG'):
        predicate = self.get_arrays_avg(temp)
      elif(method == 'MAX'):
        predicate = max(temp, key=operator.methodcaller('tolist'))
      elif(method == 'MIN'):
        predicate = min(temp, key=operator.methodcaller('tolist'))
      elif(method == 'CONCATENATE'):
        predicate = min(temp, key=operator.methodcaller('tolist'))
        maximum = max(temp, key=operator.methodcaller('tolist'))
        predicate = np.append(predicate, maximum)

      dict[example[0].rstrip()] = predicate
    return dict

  def get_cosine_similarities(self, source, target):
    """
        Calculate cosine similarity of embedded arrays
        for all possible pairs (source, target)

        Args:
            source(array): all word embeddings from the source dataset
            target(array): all word embeddings from the target dataset
       Returns:
            a pandas dataframe containing every pair (source, target) similarity
    """

    similarity = {}
    for s in source:
      for t in target:
        key = s + ',' + t
        similarity[key] = 1 - spatial.distance.cosine(source[s], target[t])

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df

  def similarity_word2vec(self, source, target, model_path, method):
    """
        Embed relations using pre-trained word2vec

        Args:
            source(array): all predicates from source dataset
            target(array): all predicates from target dataset
            model_path(str): current path to find the model to be used
            method(str): method used to compact arrays
       Returns:
             a pandas dataframe containing every pair (source, target) similarity
    """

    # Load Google's pre-trained Word2Vec model.
    word2vecModel = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')

    source = utils.build_triples(source)
    target = utils.build_triples(target)

    source = self.build_model_array(source, word2vecModel, method=method)
    target = self.build_model_array(target, word2vecModel, method=method)

    return self.get_cosine_similarities(source, target)

  def write_to_file_closest_distance(self, from_predicate, to_predicate, source, similarity, recursion=False, searchArgPermutation=False, searchEmpty=False, allowSameTargetMap=False):
    """
          Sorts dataframe to obtain the closest target to a given source

          Args:
              source(array): all predicates from source dataset
              similarity(dataframe): a pandas dataframe containing every pair (source, target) similarity
         Returns:
               writes a file containing transfer information
      """
   
    with open('transfer_file.txt', 'w') as file:
      for s in source:
        pairs = similarity.filter(like=s, axis=0).sort_values(by='similarity', ascending=False).head(10).index.tolist()
        file.write(str(s) + ': ' + ','.join([pair.split(',')[1] for pair in pairs]))
        file.write('\n')
      file.write('\n')

      file.write('setMap: ' + from_predicate + ',' + to_predicate + '\n')
      if(recursion):
          file.write('setMap: recursion_' + from_predicate + '(A,B)=recursion_' + to_predicate + '(A,B).\n')
      file.write('setParam: searchArgPermutation=' + str(searchArgPermutation).lower() + '.\n')
      file.write('setParam: searchEmpty=' + str(searchEmpty).lower() + '.\n')
      file.write('setParam: allowSameTargetMap=' + str(allowSameTargetMap).lower() + '.\n')
      file.close()
