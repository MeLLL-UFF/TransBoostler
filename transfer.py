
from gensim.test.utils import datapath, get_tmpfile
from ekphrasis.classes.segmenter import Segmenter
from gensim.models import KeyedVectors, Word2Vec
from collections import OrderedDict
import parameters as params
from scipy import spatial
from tqdm import tqdm
import utils as utils
import pandas as pd
import numpy as np
import fasttext
import operator
import sys
import os
import re

class Transfer:

  def __init__(self, seg):
    self.seg = seg

  def build_fasttext_array(self, data, model, method=None):
    """
        Turn relations into a single array

        Args:
            data(array): an array containing all predicates
            model(object): fasttext embedding model
            method(str): method to compact arrays of embedded words
       Returns:
            a dictionary that the keys are the words and the values are single arrays of embeddings
    """

    dict = {}
    for example in tqdm(data):
      temp = []

      # Tokenize words of relation
      predicate = self.seg.segment(example[0])
      for word in predicate.split():
        try:
          #temp.append(model.get_word_vector(word.lower().strip()))
          temp.append(model.wv[word.lower().strip()])
        except:
          print('Word \'{}\' not present in pre-trained model'.format(word.lower().strip()))
          temp.append([0] * params.EMBEDDING_DIMENSION)

      predicate = temp.copy()
      if(method):
        predicate = utils.single_array(temp, method)

      if(len(example) > 2 and example[2] == ''):
        example.remove('')
      dict[example[0].rstrip()] = [predicate, example[1:]]
    return dict

  def build_word2vec_array(self, data, model, method=None):
    """
        Turn relations into a single array

        Args:
            data(array): an array containing all predicates
            model(object): word2vec embedding model
            method(str): method to compact arrays of embedded words
       Returns:
            a dictionary that the keys are the words and the values are single arrays of embeddings
    """

    dict = {}
    for example in tqdm(data):
      temp = []

      # Tokenize words of relation
      predicate = self.seg.segment(example[0])
      for word in predicate.split():
        try:
          temp.append(model.wv[word.lower().strip()])
        except:
          print('Word {} not present in pre-trained model'.format(word.lower().strip()))
          temp.append([0] * params.EMBEDDING_DIMENSION)
    
      predicate = temp.copy()
      if(method):
        predicate = utils.single_array(temp, method)

      if(len(example) > 2 and example[2] == ''):
        example.remove('')
      dict[example[0].rstrip()] = [predicate, example[1:]]
    return dict

  def map_predicates(self, sources, similarity, searchArgPermutation=False, allowSameTargetMap=False):
      """
        Sorts dataframe to obtain the closest target to a given source

        Args:
            source(array): all predicates from source dataset
            similarity(dataframe): a pandas dataframe containing every pair (source, target) similarity
        Returns:
            a dictionary containing all predicates mapped
      """

      target_mapped, mapping = [], {}
      indexes = similarity.index.tolist()
      
      for index in tqdm(indexes):
        index = re.split(r',\s*(?![^()]*\))', index)
        source, target = index[0].rstrip(), index[1].rstrip()

        if(source in mapping or source not in sources):
          continue

        if(allowSameTargetMap):
          mapping[source] = target
        else:
          if(target in target_mapped):
            continue
          else:
            mapping[source] = target
            target_mapped.append(target)

        if(len(mapping) == len(sources)):
          # All sources mapped to a target
          break

      # Adds source predicates to be mapped to 'empty'
      for s in sources:
        if(s not in mapping):
          mapping[s] = ''

      del indexes
      return mapping

  def write_to_file_closest_distance(self, from_predicate, to_predicate, arity, mapping, filename, recursion=False, searchArgPermutation=False, searchEmpty=False, allowSameTargetMap=False):
    """
          Sorts dataframe to obtain the closest target to a given source

          Args:
              from_predicate(str): predicate of model trained using source data
              to_predicate(str): predicate of model to be trained transfering the structure of source model
              arity(int): arity of from and to predicate
              mapping(dict): a dictionary a pair of mapping (source, target)
         Returns:
              writes a file containing transfer information
    """
    with open(params.TRANSFER_FILENAME, 'w') as file:
      for source in mapping.keys():
        if(mapping[source] != ''):
          file.write((source.replace('`', '') + ': ' +  mapping[source]).replace('`', ''))
        else:
          file.write((source.replace('`', '') + ':'))
        file.write('\n')

      if(recursion):
          file.write('recursion_' + from_predicate + '(A,B): recursion_' + to_predicate + '(A,B)\n')
      file.write('setMap:' + from_predicate + '(' + ','.join([chr(65+i) for i in range(arity)]) + ')' + ',' + to_predicate + '(' + ','.join([chr(65+i) for i in range(arity)]) + ')' + '\n')
      #file.write('setParam:searchArgPermutation=' + str(searchArgPermutation).lower() + '.\n')
      #file.write('setParam:searchEmpty=' + str(searchEmpty).lower() + '.\n')
      file.write('setParam:allowSameTargetMap=' + str(allowSameTargetMap).lower() + '.\n')

    with open(filename + '/transfer.txt', 'w') as file:
      for source in mapping.keys():
        if(mapping[source] != ''):
          file.write((source.replace('`', '') + ': ' +  mapping[source]).replace('`', ''))
        else:
          file.write((source.replace('`', '') + ':'))
        file.write('\n')

      if(recursion):
          file.write('recursion_' + from_predicate + '(A,B): recursion_' + to_predicate + '(A,B)\n')
      file.write('setMap:' + from_predicate + '(' + ','.join([chr(65+i) for i in range(arity)]) + ')' + ',' + to_predicate + '(' + ','.join([chr(65+i) for i in range(arity)]) + ')' + '\n')
      #file.write('setParam:searchArgPermutation=' + str(searchArgPermutation).lower() + '.\n')
      #file.write('setParam:searchEmpty=' + str(searchEmpty).lower() + '.\n')
      file.write('setParam:allowSameTargetMap=' + str(allowSameTargetMap).lower() + '.\n')
