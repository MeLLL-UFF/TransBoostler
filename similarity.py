
from __future__ import division
from gensim.corpora.dictionary import Dictionary
from gensim import matutils, corpora
from scipy.spatial import distance
import parameters as params
from tqdm import tqdm
import utils as utils
from pyemd import emd
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
              sent_1 = [model[word] for word in source_segmented]
              sent_2 = [model[word] for word in target_segmented]
            
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

        if(params.METHOD):
            
            dictionary = Dictionary(documents=[[source[0]], [target[0]]])
            vocab_len = len(dictionary)
            
            # Sets for faster look-up.
            sourceset1 = set(source[0])
            targetset2 = set(target[0])
            
            distance_matrix = self.get_distance_matrix(source[0], target[0], model, dictionary, vocab_len)
            
            # Compute nBOW representation of documents
            d1 = self.nbow(source[0], dictionary, vocab_len)
            d2 = self.nbow(target[0], dictionary, vocab_len)

            # Compute WMD
            similarity[key] = emd(d1, d2, distance_matrix)
        else:
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
    

  def nbow(self, document, dictionary, vocab_len):
    """
    	Return nBoW representation of a document using a dictionary of words 

	    Args:
	        document(array): one given predicate
	        dictionary(Dictionary): pair (source, target)
	    Returns:
	        a array of frequencies/size_of_document
	"""
    d = np.zeros(vocab_len, dtype=np.double)
    nbow = dictionary.doc2bow([document])  # Word frequencies.
    doc_len = len([document])
    for idx, freq in nbow:
        d[idx] = freq / float(doc_len)  # Normalized word frequencies.
    return d

  def get_distance_matrix(self, source, target, model, dictionary, vocab_len):
      """
    	Compute distance matrix between the predicates

	    Args:
	        source(str): source predicate
	        target(str): target predicate
	        model(KeyedVectors): embedding pre-trained model
	        vocab_len(int): size of the vocabulary
	    Returns:
	        a array of distancies
	"""
      
    # Compute distance matrix.
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if not t1 in source or not t2 in target:
                continue
            
            t1_conc = np.concatenate([model[token] for token in self.seg.segment(t1).split()])
            t2_conc = np.concatenate([model[token] for token in self.seg.segment(t2).split()])
            
            if(len(t1_conc) != len(t2_conc)):
                t1_conc, t2_conc = utils.set_to_same_size(t1_conc, t2_conc, params.EMBEDDING_DIMENSION)
            
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = np.sqrt(np.sum((t1_conc - t2_conc)**2))

    if np.sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        print('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')
    return distance_matrix

#from ekphrasis.classes.segmenter import Segmenter
#from gensim.models import KeyedVectors, FastText
#from pyemd import emd

# Segmenter using the word statistics from Wikipedia
#seg = Segmenter(corpus="english")

#model = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

#document1 = ['testexample']
#document2 = ['examblood']

#document1_vec = np.array([model.wv[token] for token in document1 if token in model]).mean(axis=0)
#document2_vec = np.array([model.wv[token] for token in document2 if token in model]).mean(axis=0)


