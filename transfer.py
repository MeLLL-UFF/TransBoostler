
from gensim.test.utils import datapath, get_tmpfile
from ekphrasis.classes.segmenter import Segmenter
from gensim.models import KeyedVectors, Word2Vec
from collections import OrderedDict
from similarity import Similarity
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

  def __init__(self, segmenter, model, model_name):
    self.seg = segmenter
    self.model = model
    self.model_name = model_name
    self.similarity = Similarity(self.seg)

  def __is_compatible(self, source, target, constraints):
    """
        Turn relations into a single array

        Args:
            source(str): source predicate
            target(str): target predicate
            constraints(dict): dictionary mapping most similar literals 
       Returns:
            False if literals are not compatible
            True if literals are compatible
    """
    source_literals, target_literals = utils.get_all_literals([source]), utils.get_all_literals([target])
    for source_literal, target_literal in zip(source_literals, target_literals):
        if(constraints[source_literal] != target_literal):
            return False
    return True

  def __build_fasttext_array(self, data, method=None, literals_mapping=False):
    """
        Turn relations into a single array

        Args:
            data(array): an array containing predicates for each tree
            model(object): fasttext embedding model
            method(str): method to compact arrays of embedded words
       Returns:
            a dictionary that the keys are the words and the values are single arrays of embeddings
    """

    dict = {}
    for example in tqdm(data):
      temp = []

      try:
          # Tokenize words of relation
          predicate = self.seg.segment(example[0])
      except:
          predicate = self.seg.segment(example)

      for word in predicate.split():
        try:
          #temp.append(model.get_word_vector(word.lower().strip()))
          temp.append(self.model.wv[word.lower().strip()])
        except:
          print('Word \'{}\' not present in pre-trained model'.format(word.lower().strip()))
          temp.append([0] * params.EMBEDDING_DIMENSION)

      predicate = temp.copy()
      if(method):
        predicate = utils.single_array(temp, method)

      if(len(example) > 2 and example[2] == ''):
        example.remove('')

      if(literals_mapping):
        dict[example.rstrip()] = [predicate]
      else:
        dict[example[0].rstrip()] = [predicate, example[1:]]
    return dict

  def __build_word2vec_array(self, data, method=None, literals_mapping=False):
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

      if(isinstance(example, list)):
          # Tokenize words of relation
          predicate = self.seg.segment(example[0])
      else:
          # Literals come as single elements of lists
          predicate = self.seg.segment(example)

      for word in predicate.split():
        try:
          #temp.append(model.get_word_vector(word.lower().strip()))
          temp.append(self.model.wv[word.lower().strip()])
        except:
          print('Word \'{}\' not present in pre-trained model'.format(word.lower().strip()))
          temp.append([0] * params.EMBEDDING_DIMENSION)

      predicate = temp.copy()
      if(method):
        predicate = utils.single_array(temp, method)

      if(len(example) > 2 and example[2] == ''):
        example.remove('')
        
      if(isinstance(example, str)):
        dict[example.rstrip()] = [predicate, '']
      else:
        dict[example[0].rstrip()] = [predicate, example[1:]]
    return dict

  def __build_word_vectors(self, data, similarity_metric):
    """
        Create word vectors if needed (given the similarity metric)

        Args:
            data(list): all triples of predicates
        Returns:
            a dictionary of word vectors or a list of strings
    """
    if(similarity_metric in params.WORD_VECTOR_SIMILARITIES):
      if(self.model_name == params.FASTTEXT):
        return self.__build_fasttext_array(data, method=params.METHOD, literals_mapping=params.USE_LITERALS)
      else:
        return self.__build_word2vec_array(data, method=params.METHOD, literals_mapping=params.USE_LITERALS)
    return data

  def __create_constraints(self, sources, similarity, searchArgPermutation=False, allowSameTargetMap=False):
      """
        Create constraints to predicate mapping

        Args:
            source(array): all predicates from source dataset
            similarity(dataframe): a pandas dataframe containing every pair (source, target) similarity
        Returns:
            a dictionary containing all predicates mapped
      """

      mapped, mapping = [], {}
      indexes = similarity.index.tolist()
      
      for index in tqdm(indexes):
        source, target = index.split(',')[0].rstrip(), index.split(',')[1].rstrip()

        if(source in mapping or source not in sources):
          continue

        if(target in mapped):
          continue
        else:
          mapping[source] = target
          mapped.append(target)

        if(len(mapping) == len(sources)):
          # All sources mapped to a target
          break
      return mapping


  def __find_best_mapping(self, clause, targets, similarity_metric, targets_taken=[], constraints=[], allowSameTargetMap=False):
    """
        Calculate pairs similarity and sorts dataframe to obtain the closest target to a given source

        Args:
            clause(str): source clause
            targets(list): all targets found in the target domain
            similarity_metric(str): similarity metric to be applied
            targets_taken(list): forbidden target-predicates
        Returns:
            the closest target-predicate to the given source
      """
    source = self.__build_word_vectors([utils.build_triple(clause)], similarity_metric)
    similarities = self.similarity.compute_similarities(source, targets, similarity_metric, self.model, self.model_name)
    similarities.to_csv('{}_{}_similarities.csv'.format(clause, similarity_metric))

    targets_taken = {}
    best_mapping = ''
    indexes = similarities.index.tolist()

    for index in tqdm(indexes):
      index = re.split(r',\s*(?![^()]*\))', index)
      source, target = index[0].rstrip(), index[1].rstrip()

      # Literals must match
      if(constraints and not self.__is_compatible(source, target, constraints)):
        continue
      
      if(allowSameTargetMap):
        return target, targets_taken
      else:
        if(target in targets_taken):
          continue
        else:
          targets_taken[target] = 0
          best_mapping = target
      
    return best_mapping, targets_taken

  def map_literals(self, similarity_metric, preds_learned, targets):
    """
      Create mappings from literals found in source domain to literals found in target domain

      Args:
          similarity_metric(str): similarity metric to be applied
          preds_learned(list): all predicates learned from source domain
          target(list): all predicates found in target domain
      Returns:
          all sources mapped to the its closest target-predicate
    """

    source_literals = utils.get_all_literals(preds_learned)
    target_literals = utils.get_all_literals(targets)

    sources = self.__build_word_vectors(source_literals, similarity_metric)
    targets = self.__build_word_vectors(target_literals, similarity_metric)

    similarities = self.similarity.compute_similarities(sources, targets, similarity_metric, self.model, self.model_name)

    constraints, literals_taken = {}, {}
    indexes = similarities.index.tolist()

    for index in tqdm(indexes):
        index = re.split(r',\s*(?![^()]*\))', index)
        source_literal, target_literal = index[0].rstrip(), index[1].rstrip()

        if(source_literal in constraints or source_literal not in sources):
          continue

        if(target_literal in literals_taken):
          continue
        else:
          constraints[source_literal] = target_literal
          literals_taken[target_literal] = 0

        if(len(constraints) == len(sources)):
          # All source literals mapped to a target literal
          break

      # Adds source predicates to be mapped to 'empty'
      #for s in sources:
      #  if(s not in mapping):
      #    mapping[s] = ''

    del indexes
    return constraints

  def map_predicates(self, similarity_metric, trees, targets):
    """
      Create mappings from source to target predicates

      Args:
          similarity_metric(str): similarity metric to be applied
          trees(list): all clauses learned from the source domain
          targets(list): all predicates found in the target domain
      Returns:
          all sources mapped to the its closest target-predicate
    """

    targets = utils.build_triples(targets)
    targets = self.__build_word_vectors(targets, similarity_metric)

    mappings, targets_taken = {}, {}
    for tree in trees:
      for i in range(len(tree.keys())):
        #Process ith node
        clauses = re.split(r',\s*(?![^()]*\))', tree[i])
        for clause in clauses:
          if(clause.split('(')[0] not in mappings):
            mappings[clause.split('(')[0]], targets_taken = self.__find_best_mapping(clause, targets, similarity_metric, targets_taken)
    return mappings

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
