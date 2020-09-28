
import os
import sys
import time
#sys.path.append('../..')

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
    
def print_function(message):
    global experiment_title
    global nbr
    if not os.path.exists('experiments/' + experiment_title):
        os.makedirs('experiments/' + experiment_title)
    with open('experiments/' + experiment_title + '/' + str(nbr) + '_' + experiment_title + '.txt', 'a') as f:
        print(message, file=f)
        print(message)
        
start = time.time()
for experiment in experiments:

    experiment_title = experiments[experiment]['id'] + '_' + experiments[experiment]['source'] + '_' + experiments[experiment]['target']

    print_function('Starting experiment {} \n'.format(experiment_title))

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
    print('Source train neg examples: %s\n'.format(len(src_neg)))
                       
    # Learning from source dataset
    background = boostsrl,modes(modes=bk[source], [experiment[ 'predicate']], use_std_logic_variables=False, , maxTreeDepth=maxTreeDepth, nodeSize=nodeSize, numOfClauses=numOfClauses)

    model = boostsrl.train(background, src_pos, src_neg, src_facts)
    
    print('Model training time {}'.format(model.traintime()))

    print(['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(trees)])

    break