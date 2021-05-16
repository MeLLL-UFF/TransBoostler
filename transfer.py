
from gensim.test.utils import datapath, get_tmpfile
from ekphrasis.classes.segmenter import Segmenter
from gensim.models import KeyedVectors, Word2Vec
from collections import OrderedDict
from similarity import Similarity
import parameters as params
from scipy import spatial
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
    self.constraints = {}

  def __same_arity(self, source_literals, target_literals):
    """
        Check if predicates have the same arity

        Args:
            source_literals(list): source literals
            target_literals(list): target literals
       Returns:
            False if different arity
            True if same arity
    """
    return len(source_literals) == len(target_literals)

  def __is_compatible(self, source, target):
    """
        Turn relations into a single array

        Args:
            source(str): source predicate
            target(str): target predicate

       Returns:
            False if literals are not compatible
            True if literals are compatible
    """
    source_literals, target_literals = utils.get_all_literals([source]), utils.get_all_literals([target])
    if(self.__same_arity(source_literals, target_literals)):
      for source_literal, target_literal in zip(source_literals, target_literals):
        if(source_literal in self.constraints):
          if(self.constraints[source_literal] != target_literal):
            return False
        else:
          self.constraints[source_literal] = target_literal
      return True
    return False

  def __build_fasttext_array(self, data, mapping_literals=False):
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
    for example in data:
      temp = []

      try:
          # Tokenize words of relation
          predicate = self.seg.segment(example[0])
      except:
          predicate = self.seg.segment(example)

      for word in predicate.split():
        try:
          #temp.append(model.get_word_vector(word.lower().strip()))
          temp.append(self.model[word.lower().strip()])
        except:
          print('Word \'{}\' not present in pre-trained model'.format(word.lower().strip()))
          temp.append([0] * params.EMBEDDING_DIMENSION)

      predicate = temp.copy()
      if(params.METHOD):
        predicate = utils.single_array(temp, params.METHOD)

      if(mapping_literals):
        dict[example.strip()] = [predicate, []]
      else:
        dict[example[0].strip()] = [predicate, example[1]]
    return dict

  def __build_word2vec_array(self, data, mapping_literals=False):
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
    for example in data:
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
          temp.append(self.model[word.lower().strip()])
        except:
          print('Word \'{}\' not present in pre-trained model'.format(word.lower().strip()))
          temp.append([0] * params.EMBEDDING_DIMENSION)

      predicate = temp.copy()
      if(params.METHOD):
        predicate = utils.single_array(temp, params.METHOD)
        
      if(mapping_literals):
        dict[example.strip()] = [predicate, []]
      else:
        dict[example[0].strip()] = [predicate, example[1]]
    return dict

  def __build_word_vectors(self, data, similarity_metric, mapping_literals=False):
    """
        Create word vectors if needed (given the similarity metric)

        Args:
            data(list): all triples of predicates
        Returns:
            a dictionary of word vectors or a list of strings
    """
    if(similarity_metric in params.WORD_VECTOR_SIMILARITIES):
      if(self.model_name == params.FASTTEXT):
        return self.__build_fasttext_array(data, mapping_literals=mapping_literals)
      else:
        return self.__build_word2vec_array(data, mapping_literals=mapping_literals)
      raise 'In build_words_vectors: model should  be \'fasttext\' or \'word2vec\''
    return data

  def __find_best_mapping(self, clause, targets, similarity_metric, targets_taken={}, allowSameTargetMap=False):
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

    source  = self.__build_word_vectors([utils.build_triple(clause)], similarity_metric)
    
    similarities = self.similarity.compute_similarities(source, targets, similarity_metric, self.model, self.model_name)
    similarities.to_csv('experiments/similarities/{}/{}/{}_similarities.csv'.format(self.model_name, similarity_metric, clause.split('(')[0]))
    indexes = similarities.index.tolist()

    for index in indexes:
      index = re.split(r',\s*(?![^()]*\))', index)
      source, target = index[0].rstrip(), index[1].rstrip()

      # Literals must match
      if(not self.__is_compatible(source, target)):
        continue
      
      if(allowSameTargetMap):
        return target, targets_taken
      else:
        if(target in targets_taken):
          continue
        else:
          targets_taken[target] = 0
          return target, targets_taken
    return '', targets_taken

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

    source_literals = utils.remove_all_special_characters(utils.get_all_literals(preds_learned))
    target_literals = utils.remove_all_special_characters(utils.get_all_literals(targets))

    sources = self.__build_word_vectors(source_literals, similarity_metric, mapping_literals=True)
    targets = self.__build_word_vectors(target_literals, similarity_metric, mapping_literals=True)

    similarities = self.similarity.compute_similarities(sources, targets, similarity_metric, self.model, self.model_name)

    literals_taken = {}
    indexes = similarities.index.tolist()

    for index in indexes:
        index = index.split(',')
        source_literal, target_literal = index[0].rstrip(), index[1].rstrip()

        if(source_literal in self.constraints or source_literal not in sources):
          continue

        if(target_literal in literals_taken):
          continue
        else:
          self.constraints[source_literal] = target_literal
          literals_taken[target_literal] = 0

        if(len(self.constraints) == len(sources)):
          # All source literals mapped to a target literal
          break

    # Adds source predicates to be mapped to 'empty'
    for s in sources:
      if(s not in self.constraints):
        self.constraints[s] = ''

    del indexes
    return self.constraints

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
          if(clause not in mappings and 'recursion' not in clause):
            mappings[clause], targets_taken = self.__find_best_mapping(clause, targets, similarity_metric, targets_taken)
    return mappings

  def write_constraints_to_file(self, filename):
    """
          Write constraints file

          Args:
              similarity_metric(str): similarity metric
              embedding_model(str): model name
              mapping(dict): a dictionary a pair of literal mapping (source, target)
              filename(str): file path
         Returns:
              writes a file containing transfer information
    """
    with open(filename + '/constraints.txt', 'w') as file:
      for source in self.constraints.keys():
        if(self.constraints[source] != ''):
          file.write((source.replace('`', '') + ': ' +  self.constraints[source]).replace('`', ''))
        else:
          file.write((source.replace('`', '') + ':'))
        file.write('\n')

  def write_to_file_closest_distance(self, similarity_metric, model_name, from_predicate, to_predicate, arity, mapping, filename, recursion=False, searchArgPermutation=False, searchEmpty=False, allowSameTargetMap=False):
    """
          Write transfer file

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

    with open(filename + '/transfer_{}_{}.txt'.format(model_name, similarity_metric), 'w') as file:
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
