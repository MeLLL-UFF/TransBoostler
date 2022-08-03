
#import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('app.log','w'),logging.StreamHandler()])

from ekphrasis.classes.segmenter import Segmenter
from experiments import experiments, bk, setups
from gensim.models import KeyedVectors, FastText
from preprocessing import Preprocessing
from gensim.test.utils import datapath
from revision import TheoryRevision
from datasets.get_datasets import *
from similarity import Similarity
from boostsrl import boostsrl
from transfer import Transfer
import gensim.downloader as api
import parameters as params
import ensemble as ensemble
import utils as utils
import numpy as np
import random
import pickle
import spacy
import copy
import time
import sys
import os

#verbose=True
source_balanced = False
balanced = False

learn_from_source = False

revision = TheoryRevision()
segmenter = Segmenter(corpus="english")
preprocessing = Preprocessing(segmenter)

experiment_title = ''
experiment_type = 'transfer-experiments'

def main():

    # Loads model
    nlp = spacy.blank("en").from_disk(params.WIKIPEDIA_FASTTEXT_SPACY)
    wmd_instance = WMD.SpacySimilarityHook(nlp)

    for experiment in experiments:

        experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
        
        target = experiment['target']

        # Load total source dataset
        source_data = datasets.load(source, bk[source], seed=params.SEED)

        # Load total target dataset
        tar_total_data = datasets.load(target, bk[target], seed=params.SEED)

        utils.print_function('Starting experiment {} \n'.format(experiment_title), experiment_title, experiment_type)

        _id = experiment['id']
        source = experiment['source']
        target = experiment['target']
        predicate = experiment['predicate']
        to_predicate = experiment['to_predicate']
        arity = experiment['arity']

        # Get sources and targets
        sources = [s.replace('.', '').replace('+', '').replace('-', '') for s in set(bk[source]) if s.replace('`','').split('(')[0] != predicate and 'recursion_' not in s]
        targets = [t.replace('.', '').replace('+', '').replace('-', '') for t in set(bk[target]) if t.replace('`','').split('(')[0] != to_predicate and 'recursion_' not in t]

        sources = utils.build_triples(sources)
        targets = utils.build_triples(targets)

        path = params.ROOT_PATH + 'resources/' + experiment_title
        if not os.path.exists(path):
            os.mkdir(path)

        path = params.ROOT_PATH + 'resources/' + experiment_title + '/rwmd-similarities'
        if not os.path.exists(path):
            os.mkdir(path)
        
        similarity = {}
        for source in sources:
            start = time.time()
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
                    sent_1 = nlp(' '.join(preprocessing.pre_process_text(source[0])))
                    sent_2 = nlp(' '.join(preprocessing.pre_process_text(target[0])))
                
                    similarity[key] = wmd_instance.compute_similarity(sent_1, sent_2)

            df = pd.DataFrame.from_dict(similarity, orient="index", columns=['similarity'])
            df = df.rename_axis('candidates').sort_values(by=['similarity', 'candidates'])

            mapping_time = time.time() - start

            df.to_csv(path + source + '_similarities.csv', index=False)

            file = open(path + source + 'time.txt')
            file.write(mapping_time)
            file.close()



if __name__ == '__main__':
    sys.exit(main())
