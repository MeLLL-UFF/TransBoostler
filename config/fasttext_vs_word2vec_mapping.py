
from ekphrasis.classes.segmenter import Segmenter
from gensim.models import KeyedVectors, FastText
import pandas as pd

import sys
current_path = sys.path[0].split('/config')[0]
sys.path.append(current_path)

from experiments import experiments, bk, setups
from datasets.get_datasets import *
from similarity import Similarity
from transfer import Transfer
import parameters as params
import utils as utils

# Segmenter using the word statistics from Wikipedia
seg = Segmenter(corpus="english")

transfer = Transfer(seg)
similarity = Similarity(seg)

def fill_mappings(preds_learned, similarities, all_mappings):
	# Map source predicates to targets and creates transfer file
    mappings = transfer.map_predicates(preds_learned, similarities, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)

    for source in mappings:
    	all_mappings[source].append(mappings[source])
    return all_mappings

def map_predicates(experiment, model, modelname):
    experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
    print('Starting experiment {} \n'.format(experiment_title))

    _id = experiment['id']
    source = experiment['source']
    target = experiment['target']
    predicate = experiment['predicate']
    to_predicate = experiment['to_predicate']
    arity = experiment['arity']

    if target in ['twitter', 'yeast', 'nell_sports', 'nell_finances']:
        recursion = True
    else:
        recursion = False

    if not os.path.exists(current_path + '/mappings'):
        os.mkdir(current_path + '/mappings')

    if not os.path.exists(current_path + '/mappings/' + experiment_title):
        os.mkdir(current_path + '/mappings/' + experiment_title)

    structured = ' '.join(open(current_path + '/experiments/{}_{}_{}/{}'.format(_id, source, target, params.REFINE_FILENAME.split('/')[1]), 'r').readlines())
    structured = [p for p in bk[source] if p.split('(')[0] in structured]

    # Get the list of predicates from source tree
    preds, preds_learned = [], []
    preds = list(set(utils.sweep_tree(structured)))
    preds_learned = [pred.replace('.', '').replace('+', '').replace('-', '') for pred in bk[source] if pred.split('(')[0] != predicate and pred.split('(')[0] in preds and 'recursion_' not in pred]
    
    # Get targets
    targets = [t.replace('.', '').replace('+', '').replace('-', '') for t in set(bk[target]) if t.split('(')[0] != to_predicate and 'recursion_' not in t]

    all_mappings = {}
    for source in preds_learned:
    	all_mappings[source] = []

    # Build word vectors for source and target predicates
    sources = utils.build_triples(preds_learned)
    targets = utils.build_triples(targets)

    if(modelname == 'fasttext'):
    	# Build word vectors
        model_sources = transfer.build_fasttext_array(sources, model, method=params.METHOD)
        model_targets = transfer.build_fasttext_array(targets, model, method=params.METHOD)

        spacy_model = current_path + '/' + params.WIKIPEDIA_FASTTEXT_SPACY
    elif(modelname == 'word2vec'):
        model_sources = transfer.build_word2vec_array(sources, model, method=params.METHOD)
        model_targets = transfer.build_word2vec_array(targets, model, method=params.METHOD)

        spacy_model = current_path + '/' + params.GOOGLE_WORD2VEC_SPACY
    
    # Mapping using Cosine similarities
    similarities = similarity.cosine_similarities(model_sources, model_targets)
    all_mappings = fill_mappings(preds_learned, similarities, all_mappings)

    # Mapping using SoftCosine similarities
    similarities = similarity.soft_cosine_similarities(sources, targets, model)
    all_mappings = fill_mappings(preds_learned, similarities, all_mappings)

    # Mapping using Euclidean similarities
    similarities = similarity.euclidean_distance(model_sources, model_targets)
    all_mappings = fill_mappings(preds_learned, similarities, all_mappings)

    # Mapping using WMD similarities
    similarities = similarity.wmd_similarities(sources, targets, model)
    all_mappings = fill_mappings(preds_learned, similarities, all_mappings)

    # Mapping using Relaxed WMD similarities
    similarities = similarity.relaxed_wmd_similarities(sources, targets, spacy_model)
    all_mappings = fill_mappings(preds_learned, similarities, all_mappings)

    return pd.DataFrame.from_dict(all_mappings, orient="index", columns=['Cosine', 'SoftCosine', 'Euclidean', 'WMD', 'Relax-WMD'])

def main():

	# Starting by fasttext
    if not os.path.exists(current_path + '/' + params.WIKIPEDIA_FASTTEXT):
        raise ValueError("SKIP: You need to download the fasttext wikipedia model")

    model = KeyedVectors.load_word2vec_format(current_path + '/' + params.WIKIPEDIA_FASTTEXT, binary=False, unicode_errors='ignore')
    #loadedModel = None

    for experiment in experiments:
    	experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
    	fasttext_mappings = map_predicates(experiment, model, 'fasttext')
    	fasttext_mappings.to_csv(current_path + '/mappings/{}/fasttext_{}.csv'.format(experiment_title, params.METHOD))

    del fasttext_mappings, model

    # Starting by word2vec
    if not os.path.exists(current_path + '/' + params.GOOGLE_WORD2VEC):
        raise ValueError("SKIP: You need to download the google news model")

    model = KeyedVectors.load_word2vec_format(current_path + '/' + params.GOOGLE_WORD2VEC, binary=True, unicode_errors='ignore')
    #loadedModel = None

    for experiment in experiments:
    	experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
    	word2vec_mappings = map_predicates(experiment, model, 'word2vec')
    	word2vec_mappings.to_csv(current_path + '/mappings/{}/word2vec_{}.csv'.format(experiment_title, params.METHOD))

if __name__ == '__main__':
    sys.exit(main())