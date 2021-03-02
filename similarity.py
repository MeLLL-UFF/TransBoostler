
from gensim import matutils, corpora
from scipy.spatial import distance
import parameters as params
from tqdm import tqdm
import utils as utils
import pandas as pd
import numpy as np
import spacy
import wmd
import os

class Similarity:

  def __init__(self, segmenter):
  	self.seg = segmenter

  def cosine_similarities(self, source, target):
    """
        Calculate cosine similarity of embedded arrays
        for every possible pairs (source, target)

        Args:
            source(dict): all word embeddings from the source dataset
            target(dict): all word embeddings from the target dataset
       Returns:
            a pandas dataframe containing every pair (source, target) similarity
    """

    similarity = {}
    for s in tqdm(source):
      for t in tqdm(target):

      	# Predicates must have the same arity
        if(len(source[s][1]) != len(target[t][1])):
          continue

        key = s + '(' + ','.join(source[s][1]) + ')' + ',' + t + '(' + ','.join(target[t][1]) + ')'
        #key = s + '(' + ','.join([chr(65+i) for i in range(len(source[s][1]))]) + ')' + ',' + t + '(' + ','.join([chr(65+i) for i in range(len(target[t][1]))]) + ')'
        if(len(source[s][0]) != len(target[t][0])):
          source[s][0], target[t][0] = utils.set_to_same_size(source[s][0], target[t][0], params.EMBEDDING_DIMENSION)
          

        # This function corresponds to 1 - distance as presented at https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
        similarity[key] = distance.cosine(source[s][0], target[t][0])

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.sort_values(by='similarity')

  def soft_cosine_similarities(self, sources, targets, model):
    """
        Calculate soft cosine similarity of embedded arrays
        for every possible pairs (source, target)

        Args:
            sources(array): all predicates from the source dataset
            targets(array): all predicates from the target dataset
       Returns:
            a pandas dataframe containing every pair (source, target) similarity
    """
    
    # Get word vectors
    word_vectors = model.wv

    similarity = {}
    for source in tqdm(sources):
        if(len(source) > 2 and source[2] == ''): source.remove('')
        for target in tqdm(targets):

            if(len(target) > 2 and target[2] == ''): target.remove('')

      	    # Predicates must have the same arity
            if(len(source[1:]) != len(target[1:])): 
              continue

            key = source[0] + '(' + source[1] + ')' + ',' + target[0] + '(' + target[1] + ')'
            #key = s + '(' + ','.join([chr(65+i) for i in range(len(source[s][1]))]) + ')' + ',' + t + '(' + ','.join([chr(65+i) for i in range(len(target[t][1]))]) +')'

            # Tokenize (segment) the predicates into words
            # wasbornin -> was, born, in
            source_segmented = self.seg.segment(source[0]).split()
            target_segmented = self.seg.segment(target[0]).split()

            if(params.METHOD):
                
              # Calculate similarity using single vectors
              sent_1 = [model.wv[word] for word in source_segmented]
              sent_2 = [model.wv[word] for word in target_segmented]
            
              if(len(sent_1) != len(sent_2)):
                  sent_1, sent_2 = utils.add_dimension(sent_1, sent_2, params.EMBEDDING_DIMENSION)
                  
              sent_1, sent_2 = utils.single_array(sent_1, params.METHOD), utils.single_array(sent_2, params.METHOD)
                  
              similarity[key] = np.dot(matutils.unitvec(np.array(sent_1)), matutils.unitvec(np.array(sent_2)))
            else:
              similarity[key] = word_vectors.n_similarity(source_segmented, target_segmented)

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.sort_values(by='similarity', ascending=False)

  def wmd_similarities(self, sources, targets, model):
    """
        Calculate similarity of embedded arrays
        using Word Mover's Distances for all possible pairs (source, target)

        Args:
            sources(array): all word embeddings from the source dataset
            targets(array): all word embeddings from the target dataset
       Returns:
            a pandas dataframe containing every pair (source, target) similarity
    """

    similarity = {}
    for source in tqdm(sources):
      if(len(source) > 2 and source[2] == ''): source.remove('')
      for target in tqdm(targets):

        if(len(target) > 2 and target[2] == ''): target.remove('')

        # Predicates must have the same arity
        if(len(source[1:]) != len(target[1:])):
          continue

        key = source[0] + '(' + ','.join(source[1:]) + ')' + ',' + target[0] + '(' + ','.join(target[1:]) + ')'
        #key = s + '(' + ','.join([chr(65+i) for i in range(len(source[s][1]))]) + ')' + ',' + t + '(' + ','.join([chr(65+i) for i in range(len(target[t][1]))]) + ')'

        similarity[key] = model.wmdistance(self.seg.segment(source[0]).split(), self.seg.segment(target[0]).split())

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.sort_values(by='similarity')

  def relaxed_wmd_similarities(self, sources, targets, modelname):
      """
    	Calculate similarity of embedded arrays
	    using Relaxed Word Mover's Distance for all possible pairs (source, target)

	    Args:
	        sources(array): all word embeddings from the source dataset
	        targets(array): all word embeddings from the target dataset
	        modelname(str): name of the model to be loaded
	    Returns:
	        a pandas dataframe containing every pair (source, target) similarity
      """

      
      # Loads GoogleNews word2vec model
      nlp = spacy.blank("en").from_disk(modelname)
      similarity = {}
      for source in tqdm(sources):
        if(len(source) > 2 and source[2] == ''): source.remove('')
        for target in tqdm(targets):

            if(len(target) > 2 and target[2] == ''): target.remove('')

      	    # Predicates must have the same arity
            if(len(source[1:]) != len(target[1:])): 
              continue
              
            # Convert the sentences into SpaCy format.
            sent_1 = nlp(self.seg.segment(source[0]))
            sent_2 = nlp(self.seg.segment(target[0]))
            
            key = source[0] + '(' + source[1] + ')' + ',' + target[0] + '(' + target[1] + ')'
            
            similarity[key] = sent_2.similarity(sent_1)

      df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
      return df.sort_values(by='similarity', ascending=False)


  def euclidean_distance(self, source, target):
    """
    	Calculate similarity of embedded arrays
	    using Euclidean Distance for all possible pairs (source, target)

	    Args:
	        source(array): all word embeddings from the source dataset
	        target(array): all word embeddings from the target dataset
	    Returns:
	        a pandas dataframe containing every pair (source, target) similarity
	"""

    similarity = {}
    for s in tqdm(source):
      for t in tqdm(target):

      	# Predicates must have the same arity
        if(len(source[s][1]) != len(target[t][1])):
          continue

        key = s + '(' + ','.join(source[s][1]) + ')' + ',' + t + '(' + ','.join(target[t][1]) + ')'
        #key = s + '(' + ','.join([chr(65+i) for i in range(len(source[s][1]))]) + ')' + ',' + t + '(' + ','.join([chr(65+i) for i in range(len(target[t][1]))]) + ')'
        if(len(source[s][0]) != len(target[t][0])):
          source[s][0], target[t][0] = utils.fill_dimension(source[s][0], target[t][0], params.EMBEDDING_DIMENSION)

        similarity[key] = distance.euclidean(source[s][0], target[t][0])

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.sort_values(by='similarity')
