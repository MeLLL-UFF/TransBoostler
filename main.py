
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
    
    #print('Model training time {}'.format(model.traintime()))

    structured = []
    for i in range(params.TREES):
      structured.append(model.get_structured_tree(treenumber=i+1).copy())
    
    preds_learned = list(set(utils.sweep_tree(structured)))
    #preds_learned.remove(predicate)

    refine_structure = utils.get_all_rules_from_tree(structured)
    utils.write_to_file(refine_structure, params.REFINE_FILENAME)
    
    similarities = pd.read_csv('similaridades-10.csv').set_index('Unnamed: 0')
    #preds_learned = ["actor(A)", "movie(A)", "director(A,B)"]

    #similarities = transfer.similarity_word2vec(preds_learned, bk[target], params.GOOGLE_WORD2VEC_PATH, method=params.METHOD)
    transfer.write_to_file_closest_distance(predicate, to_predicate, preds_learned, similarities, searchArgPermutation=True, allowSameTargetMap=False)
    
    # Load new predicate target dataset
    tar_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=params.SEED)
    
    # Group and shuffle
    if target not in ['nell_sports', 'nell_finances', 'yago2s']:
        [tar_train_facts, tar_test_facts] =  datasets.get_kfold_small(i, tar_data[0])
        [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, tar_data[1])
        [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, tar_data[2])
    else:
        [tar_train_facts, tar_test_facts] =  [tar_data[0][0], tar_data[0][0]]
        to_folds_pos = datasets.split_into_folds(tar_data[1][0], n_folds=n_folds, seed=params.SEED)
        to_folds_neg = datasets.split_into_folds(tar_data[2][0], n_folds=n_folds, seed=params.SEED)
        [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, to_folds_pos)
        [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, to_folds_neg)
        
    print('Target train facts examples: %s' % len(tar_train_facts))
    print('Target train pos examples: %s' % len(tar_train_pos))
    print('Target train neg examples: %s\n' % len(tar_train_neg))
    print('Target test facts examples: %s' % len(tar_test_facts))
    print('Target test pos	 examples: %s' % len(tar_test_pos))
    print('Target test neg examples: %s\n' % len(tar_test_neg))

    background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)
    model = boostsrl.train(background, tar_train_pos, tar_train_neg, tar_train_facts, refine=params.REFINE_FILENAME, transfer=params.TRANSFER_FILENAME, trees=params.TREES)

    will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_structured_tree(treenumber=i+1))) for i in range(params.TREES)]
    for i in will:
        print(i)
    print('\n')

    #print('Model training time using transfer learning {}'.format(model.traintime()))

    results = boostsrl.test(model, tar_test_pos, tar_test_neg, tar_test_facts, trees=params.TREES)
    inference_time = results.testtime()
    t_results = results.summarize_results()
    t_results['Learning time'] = learning_time
    t_results['Inference time'] = inference_time
    print_function('Results')
    print_function('   AUC ROC   = %s' % t_results['AUC ROC'])
    print_function('   AUC PR    = %s' % t_results['AUC PR'])
    print_function('   CLL        = %s' % t_results['CLL'])
    print_function('   Precision = %s at threshold = %s' % (t_results['Precision'][0], t_results['Precision'][1]))
    print_function('   Recall    = %s' % t_results['Recall'])
    print_function('   F1        = %s' % t_results['F1'])
    print_function('\n')
    print_function('Total learning time: %s seconds' % learning_time)
    print_function('Total inference time: %s seconds' % inference_time)
    print_function('AUC ROC: %s' % t_results['AUC ROC'])
    print_function('\n')

    break