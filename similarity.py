
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.matutils import softcossim
import parameters as params
from gensim import corpora
from scipy import spatial
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
        if(source[s][0].shape[0] != target[t][0].shape[0]):
          source[s][0], target[t][0] = utils.fill_dimension(source[s][0], target[t][0], params.EMBEDDING_DIMENSION)

        similarity[key] = spatial.distance.cosine(source[s][0], target[t][0])

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.sort_values(by='similarity', ascending=False)

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

    # Tokenize (segment) the predicates into words
    # wasbornin -> was, born, in
    texts =  [self.seg.segment(source[0]).split() for source in sources]
    texts += [self.seg.segment(target[0]).split() for target in targets]

    # Create dictionary
    #dictionary = corpora.Dictionary(texts)

    # Prepare the similarity matrix
    #similarity_matrix = model.similarity_matrix(dictionary, tfidf=None)

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

            # Convert the sentences into bag-of-words vectors.
            #sent_1 = dictionary.doc2bow(self.seg.segment(source[0]).split())
            #sent_2 = dictionary.doc2bow(self.seg.segment(target[0]).split())
            
            #TODO:
            # Adicionando vetor
            # sent_2 = [(sent_2[0][0], 234565676767)]

            # adicionar concatenação 

            key = source[0] + '(' + source[1] + ')' + ',' + target[0] + '(' + target[1] + ')'
            #key = s + '(' + ','.join([chr(65+i) for i in range(len(source[s][1]))]) + ')' + ',' + t + '(' + ','.join([chr(65+i) for i in range(len(target[t][1]))]) +')'

            #similarity[key] = softcossim(sent_1, sent_2, similarity_matrix)
            similarity[key] = word_vectors.n_similarity(self.seg.segment(source[0]).split(), self.seg.segment(target[0]).split())

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

        similarity[key] = np.linalg.norm([s_i - t_i for s_i, t_i in zip(source[s][0], target[t][0])])

    df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
    return df.sort_values(by='similarity')


#from ekphrasis.classes.segmenter import Segmenter
#from gensim.models import KeyedVectors, FastText

#from gensim.corpora import Dictionary
#from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
#from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix

#model = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# segmenter using the word statistics from Wikipedia
#seg = Segmenter(corpus="english")

#sources = [['advisedby', 'a', 'b']]
#targets = [['workedunder', 'a', 'b'], ['movie', 'a', 'b']]

# Tokenize(segment) the predicates into words
#texts =  [[word for word in seg.segment(source[0]).split()] for source in sources]
#texts += [[word for word in seg.segment(target[0]).split()] for target in targets]

#termsim_index = WordEmbeddingSimilarityIndex(model.wv)
#dictionary = Dictionary(texts)
#bow_corpus = [dictionary.doc2bow(document) for document in texts]
#similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
#docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)

#print(docsim_index[dictionary.doc2bow(['advised', 'by'])])


#print(seg.segment(sources[0][0]), seg.segment(targets[0][0]))

#sim = Similarity(seg)
#df = sim.softcossim(sources, targets, model)
#print(df)
