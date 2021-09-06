
from __future__ import division
from gensim.corpora.dictionary import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.models import KeyedVectors, FastText
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

  def __init__(self, preprocessing, similarity_matrix):
    self.preprocessing = preprocessing
    self.similarity_matrix = similarity_matrix
  
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

  def __bow(self, source, source_vectors, target, target_vectors, dimension):
    """
        Builds a Bag-of-Words so source and target predicates have the same size

      Args:
          source(str): source predicate
          source_vectors: embedding vectors for source-predicate
          target(str): target predicate
          target_vectors: embedding vectors for target-predicate
          dimension(int): size of word vectors
      Returns:
          source and target embeddings set to the same size
    """

    words = source + target
    new_source, new_target = [[0]* dimension] * len(words), [[0]* dimension] * len(words)

    for i in range(len(words)):
      if words[i] in source and words[i] in target:
        source_index = source.index(words[i])
        target_index = target.index(words[i])
        new_source[i] = source_vectors[source_index][:]
        new_target[i] = target_vectors[target_index][:]
      elif words[i] in source:
        index = source.index(words[i])
        new_source[i] = source_vectors[index][:]
      elif words[i] in target:
        index = target.index(words[i])
        new_target[i] = target_vectors[index][:]
        
    return new_source, new_target

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
                source_segmented = self.preprocessing.pre_process_text(t1)
                target_segmented = self.preprocessing.pre_process_text(t2)
                
                sent_1 = [model[w] for w in source_segmented]
                sent_2 = [model[w] for w in target_segmented]

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

    source, target = self.preprocessing.pre_process_text(source), self.preprocessing.pre_process_text(target)
        
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

  def compute_similarities(self, source, targets, similarity_metric, model='', model_name='', similarity_index='', similarity_matrix='', dictionary=''):
    """
        Calculate similarities between a clause and the targets

        Args:
            source(str): source predicate
            target(str): target predicate
       Returns:
            a dataframe containing each pair similarity
    """

    if(similarity_metric == 'cosine'):
      return self.cosine_similarities(source, targets, model)

    if(similarity_metric == 'euclidean'):
      return self.euclidean_distance(source, targets)
            
    if(similarity_metric == 'softcosine'):
      return self.soft_cosine_similarities(source, targets)
    
    if(similarity_metric == 'wmd'):
      return self.wmd_similarities(source, targets, model)
            
    if(similarity_metric == 'relax-wmd' and model_name==params.FASTTEXT):
      return self.relaxed_wmd_similarities(source, targets, params.WIKIPEDIA_FASTTEXT_SPACY)

    if(similarity_metric == 'relax-wmd' and model_name==params.WORD2VEC):
      return self.relaxed_wmd_similarities(source, targets, params.GOOGLE_WORD2VEC_SPACY)

    if(similarity_metric == 'ensemble'):
        return self.ensemble_similarities(source, targets)

    raise "Similarity metric not implemented."

  def cosine_similarities(self, sources, targets, model):
    """
        Calculate cosine similarity of embedded arrays
        for every possible pairs (source, target)

        Args:
            sources(list): all word embeddings from the source dataset
            targets(list): all word embeddings from the target dataset
       Returns:
            a pandas dataframe containing every pair (source, target) similarity
    """
    similarity = {}
    for source in sources:
      for target in targets:

        key = self.__create_key(source, target)

        if(len(source[1]) != len(target[1])):
          continue

        if '()' in key: key = key.replace('(', '').replace(')', '')

        source_segmented = self.preprocessing.pre_process_text(source[0])
        target_segmented = self.preprocessing.pre_process_text(target[0])

        n_source = [model[word] for word in source_segmented if word in model]
        n_target = [model[word] for word in target_segmented if word in model]

        #n_source, n_target = self.__bow(source_segmented, sources[s][0], target_segmented, targets[t][0], params.EMBEDDING_DIMENSION)

        #if(params.METHOD):
        #  n_source, n_target = np.concatenate(n_source), np.concatenate(n_target)

        # This function corresponds to 1 - distance as presented at https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
        similarity[key] = np.dot(matutils.unitvec(np.array(n_source).mean(axis=0)), matutils.unitvec(np.array(n_target).mean(axis=0)))

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.rename_axis('candidates').sort_values(by=['similarity', 'candidates'], ascending=[False, True])

  def soft_cosine_similarities(self, sources, targets):
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

            if(len(source[1]) != len(target[1])):
              continue

            key = self.__create_key(source, target)
            
            sent_1 = self.preprocessing.pre_process_text(source[0])
            sent_2 = self.preprocessing.pre_process_text(target[0])

            # Prepare a dictionary and a corpus.
            documents  = [sent_1, sent_2]
            dictionary = corpora.Dictionary(documents)

            # Convert the sentences into bag-of-words vectors.
            sent_1 = dictionary.doc2bow(sent_1)
            sent_2 = dictionary.doc2bow(sent_2)

            # Compute soft cosine similarity
            similarity[key] = self.similarity_matrix.inner_product(sent_1, sent_2, normalized=(True,True))

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.rename_axis('candidates').sort_values(by=['similarity', 'candidates'], ascending=[False, True])

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

        if(len(source[1]) != len(target[1])):
          continue

        key = self.__create_key(source, target)

        similarity[key] = self.__wmdistance(source[0], target[0], model)

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.rename_axis('candidates').sort_values(by=['similarity', 'candidates'])

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

            if(len(source[1]) != len(target[1])):
                continue
          
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
                sent_1 = nlp(' '.join(self.preprocessing.pre_process_text(source[0])))
                sent_2 = nlp(' '.join(self.preprocessing.pre_process_text(target[0])))
            
                similarity[key] = wmd_instance.compute_similarity(sent_1, sent_2)

      df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
      return df.rename_axis('candidates').sort_values(by=['similarity', 'candidates'])

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

        if(len(sources[s][1]) != len(targets[t][1])):
          continue

        key = self.__create_key([s, sources[s][1]], [t, targets[t][1]])

        if '()' in key: key = key.replace('(', '').replace(')', '')

        source_segmented = self.preprocessing.pre_process_text(s)
        target_segmented = self.preprocessing.pre_process_text(t)

        n_source, n_target = sources[s][0], targets[t][0]
        if(len(source_segmented) != len(target_segmented)):
          n_source, n_target = self.__bow(source_segmented, sources[s][0], target_segmented, targets[t][0], params.EMBEDDING_DIMENSION)

        n_source, n_target = np.concatenate(n_source), np.concatenate(n_target)

        similarity[key] = distance.euclidean(n_source, n_target)

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.rename_axis('candidates').sort_values(by=['similarity', 'candidates'])

# from ekphrasis.classes.segmenter import Segmenter
# from pyemd import emd

# # Segmenter using the word statistics from Wikipedia
# seg = Segmenter(corpus="english")

# # fraseA = 'obama speaks media illinois'
# # fraseB = 'president greets press chicago'

# sent_1 = [['Dravid is a cricket player and a opening batsman', ['A']]]
# sent_2 = [['Leo is a cricket player too He is a batsman,baller and keeper', ['B']]]

# model = KeyedVectors.load_word2vec_format('resources/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# preprocessing = Preprocessing(seg)
# sim = Similarity(preprocessing)

# a = sim.soft_cosine_similarities(sent_1, sent_2, model)

# print(a)

# #print(sim.wmd_similarities([['Obama speaks to the media in Illinois', 'person', 'person']], [['The president greets the press in Chicago', 'person', 'person']], model))

# print(sim.wmd_similarities([[''.join(fraseA), 'person', 'person']], [[''.join(fraseB), 'person', 'person']], model))
# print(model.wmdistance(''.join(fraseA), ''.join(fraseB)))
