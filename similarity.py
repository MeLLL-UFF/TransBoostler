
from __future__ import division
from gensim.corpora.dictionary import Dictionary
from preprocessing import Preprocessing
from gensim import matutils, corpora
from scipy.spatial import distance
import parameters as params
import utils as utils
from pyemd import emd
import pandas as pd
from wmd import WMD
import numpy as np
import spacy
import wmd
import os

class Similarity:

  def __init__(self, preprocessing):
    self.preprocessing = preprocessing
  
  def __nbow(self, document, dictionary, vocab_len):
    """
      nBoW representation of a document using a dictionary of words 

      Args:
          document(array): one given predicate
          dictionary(Dictionary): pair (source, target)
      Returns:
          a list of frequencies/size_of_document
    """
    d = np.zeros(vocab_len, dtype=np.double)
    nbow = dictionary.doc2bow(document)  # Word frequencies.
    doc_len = len(document)
    for idx, freq in nbow:
        d[idx] = freq / float(doc_len)  # Normalized word frequencies.
    return d

  def __get_distance_matrix(self, source, target, model, dictionary, vocab_len):
    """
        Compute distance matrix between the predicates

	    Args:
	        source(str): source predicate
	        target(str): target predicate
	        model(KeyedVectors): embedding pre-trained model
	        vocab_len(int): size of the vocabulary
	    Returns:
	        a list of distancies
	"""
	
	  # Sets for faster look-up.
    docset1 = set(source)
    docset2 = set(target)

    # Compute distance matrix.
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if not t1 in docset1 or not t2 in docset2:
                continue

            if(params.METHOD):
                # Concatenate word vectors before calculate Euclidean Distance
                _t1, _t2 = np.concatenate([model[w] for w in self.preprocessing.pre_process_text(t1)]),np.concatenate([model[w] for w in self.preprocessing.pre_process_text(t2)])
                
                if(len(_t1) != len(_t2)):
                    _t1, _t2 = utils.set_to_same_size(_t1, _t2, params.EMBEDDING_DIMENSION)
            else:
                _t1, _t2 = model[t1], model[t2]
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = np.sqrt(np.sum((_t1 - _t2)**2))

    if np.sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        print('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')
    return distance_matrix

  def __wmdistance(self, source, target, model):
    """
        Calculate Word Mover's Distance for single-vectors (concatenate)

        Args:
            source(str): source predicate
            target(str): target predicate
       Returns:
            distance between two word vectors
    """
    if(params.METHOD):
        source, target = [source], [target]
    else:
        source, target = self.preprocessing.pre_process_text(source), self.preprocessing.pre_process_text(target)
        #source, target = source.split(), target.split()
        
    dictionary = Dictionary(documents=[source, target])
    vocab_len = len(dictionary)

    # Source and target can be of the same type, distance_matrix must have at least two unique tokens
    if(len(dictionary) == 1):
      return 1.0
        
    distance_matrix = self.__get_distance_matrix(source, target, model, dictionary, vocab_len)
    
     # Compute nBOW representation of documents
    d1 = self.__nbow(source, dictionary, vocab_len)
    d2 = self.__nbow(target, dictionary, vocab_len)

    # Compute WMD
    return emd(d1, d2, distance_matrix)

  def __spacy_nbow(self, texts, nlp):
    """
        Calculates SpaCy nbow model

        Args:
            texts(list): source predicate
            nlp(spaCy): the SpaCy embedding model
       Returns:
            a dictionary containing the nBoW model
    """
    documents = {}
    for text in texts:
        text = nlp(text)
        tokens = [t for t in text if t.is_alpha and not t.is_stop]
        words = Counter(t.text for t in tokens)
        orths = {t.text: t.orth for t in tokens}
        sorted_words = sorted(words)
        documents[text] = (text, [orths[t] for t in sorted_words],
                        np.array([words[t] for t in sorted_words],
                                    dtype=np.float32))
    return documents

  def __create_key(self, source, target):
    """
        Create key to to dataframe used for mapping

        Args:
            source(list/str): source predicate and its types
            target(list/str): target predicate and its types
       Returns:
            a string corresponding that corresponds to the mapping
    """

    return source[0] + '(' + ','.join(source[1]) + ')' + ',' + target[0] + '(' + ','.join(target[1]) + ')'
    #key = s + '(' + ','.join([chr(65+i) for i in range(len(source[s][1]))]) + ')' + ',' + t + '(' + ','.join([chr(65+i) for i in range(len(target[t][1]))]) + ')'

  def compute_similarities(self, source, targets, similarity_metric, model='', model_name=''):
    """
        Calculate similarities between a clause and the targets

        Args:
            source(str): source predicate
            target(str): target predicate
       Returns:
            a dataframe containing each pair similarity
    """

    if(similarity_metric == 'cosine'):
      return self.cosine_similarities(source, targets)

    if(similarity_metric == 'euclidean'):
      return self.euclidean_distance(source, targets)
            
    if(similarity_metric == 'softcosine'):
      return self.soft_cosine_similarities(source, targets, model)
    
    if(similarity_metric == 'wmd'):
      return self.wmd_similarities(source, targets, model)
            
    if(similarity_metric == 'relax-wmd' and model_name==params.FASTTEXT):
      return self.relaxed_wmd_similarities(source, targets, params.WIKIPEDIA_FASTTEXT_SPACY)

    if(similarity_metric == 'relax-wmd' and model_name==params.WORD2VEC):
      return self.relaxed_wmd_similarities(source, targets, params.GOOGLE_WORD2VEC_SPACY)

    raise "Similarity metric not implemented."

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
    for s in source:
      for t in target:

        key = self.__create_key([s, source[s][1]], [t, target[t][1]])

        if '()' in key: key = key.replace('(', '').replace(')', '')

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

    similarity = {}
    for source in sources:
        for target in targets:

            key = self.__create_key(source, target)
            
            source_segmented = self.preprocessing.pre_process_text(source[0])
            target_segmented = self.preprocessing.pre_process_text(target[0])
                
            # Calculate similarity using single vectors
            sent_1 = [model[word] for word in source_segmented if word in model]
            sent_2 = [model[word] for word in target_segmented if word in model]
            
            if(params.METHOD):
                sent_1, sent_2 = np.concatenate(sent_1), np.concatenate(sent_2)
                
                if(len(sent_1) != len(sent_2)):
                    sent_1, sent_2 = utils.set_to_same_size(sent_1, sent_2, params.EMBEDDING_DIMENSION)
                
                similarity[key] = np.dot(matutils.unitvec(np.array(sent_1)), matutils.unitvec(np.array(sent_2)))
            else:
                similarity[key] = np.dot(matutils.unitvec(np.array(sent_1).mean(axis=0)), matutils.unitvec(np.array(sent_2).mean(axis=0)))
                
            #similarity[key] = word_vectors.n_similarity(source_segmented, target_segmented)

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
    for source in sources:
      for target in targets:

        key = self.__create_key(source, target)

        similarity[key] = self.__wmdistance(source[0], target[0], model)

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
      wmd_instance = WMD.SpacySimilarityHook(nlp)

      similarity = {}
      for source in sources:
        for target in targets:
          
            key = self.__create_key(source, target)
          
            if(params.METHOD):
                words = set([source[0]]).union([target[0]])
                embeddings = [np.concatenate([nlp.vocab[w].vector for w in self.preprocessing.pre_process_text(word)]) for word in words]
                
                #embeddings = [np.concatenate([nlp.vocab[w].vector for w in self.seg.segment(source[0]).split()]),np.concatenate([nlp.vocab[w].vector for w in self.seg.segment(target[0]).split()])]

                if(len(embeddings) > 1 and len(embeddings[0]) != len(embeddings[1])):
                    embeddings[0], embeddings[1] = utils.set_to_same_size(embeddings[0], embeddings[1], params.EMBEDDING_DIMENSION)

                similarity[key] = wmd_instance.compute_similarity(nlp(source[0]), nlp(target[0]), evec=np.array(embeddings, dtype=np.float32), single_vector=True)
            else:
                # Convert the sentences into SpaCy format.
                sent_1 = nlp(self.seg.segment(source[0]))
                sent_2 = nlp(self.seg.segment(target[0]))
            
                similarity[key] = wmd_instance.compute_similarity(sent_1, sent_2)

      df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
      return df.sort_values(by='similarity', ascending=True)

  def euclidean_distance(self, sources, targets):
    """
    	Calculate similarity of embedded arrays
	    using Euclidean Distance for all possible pairs (source, target)

	    Args:
	        sources(dict): all word embeddings from the source dataset
	        targets(list): all word embeddings from the target dataset
	    Returns:
	        a pandas dataframe containing every pair (source, target) similarity
    """
    similarity = {}
    for s in sources:
      for t in targets:

        key = self.__create_key([s, sources[s][1]], [t, targets[t][1]])

        if '()' in key: key = key.replace('(', '').replace(')', '')

        if(len(sources[s][0]) != len(targets[t][0])):
          sources[s][0], targets[t][0] = utils.set_to_same_size(sources[s][0], targets[t][0], params.EMBEDDING_DIMENSION)

        similarity[key] = distance.euclidean(sources[s][0], targets[t][0])

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.sort_values(by='similarity')

# from ekphrasis.classes.segmenter import Segmenter
# from gensim.models import KeyedVectors, FastText
# from pyemd import emd

# # Segmenter using the word statistics from Wikipedia
# seg = Segmenter(corpus="english")

# fraseA = 'obama speaks media illinois'
# fraseB = 'president greets press chicago'

# model = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# sim = Similarity(seg)
# #print(sim.wmd_similarities([['Obama speaks to the media in Illinois', 'person', 'person']], [['The president greets the press in Chicago', 'person', 'person']], model))
# #params.METHOD = None
# print(sim.wmd_similarities([[''.join(fraseA), 'person', 'person']], [[''.join(fraseB), 'person', 'person']], model))
# print(model.wmdistance(''.join(fraseA), ''.join(fraseB)))
