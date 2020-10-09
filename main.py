
import os
import sys
import time

from sklearn.model_selection import train_test_split
from experiments import experiments, bk
from datasets.get_datasets import *
from boostsrl import boostsrl
from transfer import Transfer
import parameters as params
import utils as utils
import numpy as np
import random

#verbose=True
source_balanced = 1
balanced = 1

transfer = Transfer()

if not os.path.exists('experiments'):
    os.makedirs('experiments')
      
start = time.time()
for experiment in experiments:

    experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']

    print('Starting experiment {} \n'.format(experiment_title))

    source = experiment['source']
    target = experiment['target']
    predicate = experiment['predicate']
    to_predicate = experiment['to_predicate']
    
    # Load source dataset
    src_total_data = datasets.load(source, bk[source], seed=params.SEED)
    src_data = datasets.load(source, bk[source], target=predicate, balanced=source_balanced, seed=params.SEED)

    # Group and shuffle
    src_facts = datasets.group_folds(src_data[0])
    src_pos = datasets.group_folds(src_data[1])
    src_neg = datasets.group_folds(src_data[2])
                
    print('Start learning from source dataset\n')
    
    print('Source train facts examples: {}'.format(len(src_facts)))
    print('Source train pos examples: {}'.format(len(src_pos)))
    print('Source train neg examples: {}\n'.format(len(src_neg)))
    
    # Learning from source dataset
    background = boostsrl.modes(bk[source], [experiment['predicate']], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)

    model = boostsrl.train(background, src_pos, src_neg, src_facts, refine=params.REFINE, trees=params.TREES)
    
    print('Model training time {}'.format(model.traintime()))

    structured = []
    for i in range(params.TREES):
      structured.append(model.get_structured_tree(treenumber=i+1).copy())
    
    preds_learned = list(set(utils.sweep_tree(structured)))
    preds_learned.remove(predicate)

    #similarities = pd.read_csv('similaridades-10.csv').set_index('Unnamed: 0')
    #preds_learned = ["actor", "movie", "director"]

    similarities = transfer.similarity_word2vec(preds_learned, bk[target].remove(to_predicate), params.GOOGLE_WORD2VEC_PATH, method=params.METHOD)
    transfer.write_to_file_closest_distance(predicate, to_predicate, preds_learned, similarities, searchArgPermutation=True, allowSameTargetMap=False)
      
    break

    