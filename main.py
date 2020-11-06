
from experiments import experiments, bk
from datasets.get_datasets import *
from boostsrl import boostsrl
from transfer import Transfer
import parameters as params
import utils as utils
import numpy as np
import random
import time
import sys
import os

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

    _id = experiment['id']
    source = experiment['source']
    target = experiment['target']
    predicate = experiment['predicate']
    to_predicate = experiment['to_predicate']
    arity = experiment['arity']

    path = os.getcwd() + '/experiments/{}_{}_{}'.format(_id, source, target)
    if not os.path.exists(path):
        os.mkdir(path)
    
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
    
    #TODO: adicionar o tempo corretamente
    #print('Model training time {}'.format(model.traintime()))

    structured = []
    for i in range(params.TREES):
      structured.append(model.get_structured_tree(treenumber=i+1).copy())
    
    # Get the list of predicates from source tree
    preds = list(set(utils.sweep_tree(structured)))
    preds_learned = list(set([pred.replace('.', '').replace('+', '').replace('-', '') for pred in bk[source] if pred.split('(')[0] != predicate and pred.split('(')[0] in preds]))
    
    refine_structure = utils.get_all_rules_from_tree(structured)
    utils.write_to_file(refine_structure, params.REFINE_FILENAME)
    utils.write_to_file(refine_structure, os.getcwd() + '/experiments/{}_{}_{}/'.format(_id, source, target) + 'source_tree.txt')

    similarities = transfer.similarity_fasttext(preds_learned, set(bk[target]), params.WIKIPEDIA_FASTTEXT_PATH, method=params.METHOD)
    transfer.write_to_file_closest_distance(predicate, to_predicate, arity, set([s.replace('.', '').replace('+', '').replace('-', '') for s in bk[source] if s.replace('.', '').replace('+', '').replace('-', '') in preds_learned]), similarities, allowSameTargetMap=True)
    
    # Load new predicate target dataset
    tar_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=params.SEED)
    
    # Group and shuffle
    i = 0
    n_folds = 1
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
    
    print('Start transfer learning experiment\n')

    print('Target train facts examples: %s' % len(tar_train_facts))
    print('Target train pos examples: %s' % len(tar_train_pos))
    print('Target train neg examples: %s\n' % len(tar_train_neg))
    print('Target test facts examples: %s' % len(tar_test_facts))
    print('Target test pos	 examples: %s' % len(tar_test_pos))
    print('Target test neg examples: %s\n' % len(tar_test_neg))

    start = time.time()

    background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)
    model = boostsrl.train(background, tar_train_pos, tar_train_neg, tar_train_facts, refine=params.REFINE_FILENAME, transfer=params.TRANSFER_FILENAME, trees=params.TREES)
    
    end = time.time()
    learning_time = end-start

    #print('Model training time using transfer learning {}'.format(model.traintime()))

    start = time.time()

    results = boostsrl.test(model, tar_test_pos, tar_test_neg, tar_test_facts, trees=params.TREES)

    end = time.time()
    inference_time = end-start

    inference_time = results.testtime()
    t_results = results.summarize_results()
    results = []
    t_results['Learning time'] = learning_time
    t_results['Inference time'] = inference_time
    results.append('Results')
    results.append('   AUC ROC   = {}'.format(t_results['AUC ROC']))
    results.append('   AUC PR    = {}'.format(t_results['AUC PR']))
    results.append('   CLL       = {}'.format(t_results['CLL']))
    results.append('   Precision = {} at threshold = {}'.format(t_results['Precision'][0], t_results['Precision'][1]))
    results.append('   Recall    = {}'.format(t_results['Recall']))
    results.append('   F1        = {}'.format(t_results['F1']))
    results.append('\n')
    results.append('Total learning time: {} seconds'.format(learning_time))
    results.append('Total inference time: {} seconds'.format(inference_time))
    results.append('AUC ROC: {}'.format(t_results['AUC ROC']))
    results.append('\n')

    structured = []
    for i in range(params.TREES):
      structured.append(model.get_structured_tree(treenumber=i+1).copy())

    refine_structure = utils.get_all_rules_from_tree(structured)

    results += refine_structure
    utils.write_to_file(results, os.getcwd() + '/experiments/{}_{}_{}/results.txt'.format(_id, source, target))