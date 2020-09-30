
import os
import sys
import time
#sys.path.append('../..')

from sklearn.model_selection import train_test_split
from experiments import experiments, bk
from datasets.get_datasets import *
from boostsrl import boostsrl
import numpy as np
import random

#verbose=True
source_balanced = 1
balanced = 1
firstRun = False
n_runs = 24
folds = 3

nodeSize = 2
numOfClauses = 8
maxTreeDepth = 3
trees = 10
seed = 441773

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
    src_total_data = datasets.load(source, bk[source], seed=seed)
    src_data = datasets.load(source, bk[source], target=predicate, balanced=source_balanced, seed=seed)

    # Group and shuffle
    src_facts = datasets.group_folds(src_data[0])
    src_pos = datasets.group_folds(src_data[1])
    src_neg = datasets.group_folds(src_data[2])
                
    print('Start learning from source dataset\n')
    
    print('Source train facts examples: {}'.format(len(src_facts)))
    print('Source train pos examples: {}'.format(len(src_pos)))
    print('Source train neg examples: {}\n'.format(len(src_neg)))
    
    start = time.time()
    # Learning from source dataset
    background = boostsrl.modes(bk[source], [experiment['predicate']], useStdLogicVariables=False, maxTreeDepth=maxTreeDepth, nodeSize=nodeSize, numOfClauses=numOfClauses)

    model = boostsrl.train(background, src_pos, src_neg, src_facts, refine=None, trees=10)

    end = time.time()
    
    print('Model training time {}'.format(round(end-start, 4)))

    #results = boostsrl.test(model, src_pos, src_neg, src_facts, trees=10)
    #print(results.summarize_results())

    structured = []
    for i in range(trees):
      structured.append(model.get_structured_tree(treenumber=i+1).copy())
    print(structured)

    break