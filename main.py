
from experiments import experiments, bk
from datasets.get_datasets import *
from boostsrl import boostsrl
from transfer import Transfer
import parameters as params
from gensim.models import KeyedVectors
import utils as utils
import numpy as np
import fasttext
import random
import time
import sys
import os
import gc

import logging
logging.basicConfig(level=logging.INFO)

fastTextModel = fasttext.load_model(params.WIKIPEDIA_FASTTEXT_PATH)
#word2vecModel = KeyedVectors.load_word2vec_format(params.GOOGLE_WORD2VEC_PATH, binary=True)

#verbose=True
source_balanced = 1
balanced = 1

transfer = Transfer()

if not os.path.exists('experiments'):
    os.makedirs('experiments')
      
for experiment in experiments:

    experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']

    logging.info('Starting experiment {} \n'.format(experiment_title))

    _id = experiment['id']
    source = experiment['source']
    target = experiment['target']
    predicate = experiment['predicate']
    to_predicate = experiment['to_predicate']
    arity = experiment['arity']

    path = os.getcwd() + '/experiments/' + experiment_title
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Load source dataset
    src_total_data = datasets.load(source, bk[source], seed=params.SEED)
    src_data = datasets.load(source, bk[source], target=predicate, balanced=source_balanced, seed=params.SEED)

    # Group and shuffle
    src_facts = datasets.group_folds(src_data[0])
    src_pos   = datasets.group_folds(src_data[1])
    src_neg   = datasets.group_folds(src_data[2])
                
    logging.info('Start learning from source dataset\n')
    
    logging.info('Source train facts examples: {}'.format(len(src_facts)))
    logging.info('Source train pos examples: {}'.format(len(src_pos)))
    logging.info('Source train neg examples: {}\n'.format(len(src_neg)))
    
    start = time.time()

    # Learning from source dataset
    background = boostsrl.modes(bk[source], [experiment['predicate']], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)
    model = boostsrl.train(background, src_pos, src_neg, src_facts, trees=params.TREES)
    
    end = time.time()

    #TODO: adicionar o tempo corretamente
    #print('Model training time {}'.format(model.traintime()))

    logging.info('Model training time {}'.format(end-start))

    logging.info('Building refine structure')

    # Get all learned trees
    structured = []
    for i in range(params.TREES):
      structured.append(model.get_structured_tree(treenumber=i+1).copy())
    
    # Get the list of predicates from source tree
    preds, preds_learned = [], []
    preds = list(set(utils.sweep_tree(structured)))
    preds_learned = [pred.replace('.', '').replace('+', '').replace('-', '') for pred in bk[source] if pred.split('(')[0] != predicate and pred.split('(')[0] in preds]

    logging.info('Searching for similarities')

    # Get all rules learned by RDN-B
    refine_structure = utils.get_all_rules_from_tree(structured)
    utils.write_to_file(refine_structure, params.REFINE_FILENAME)
    utils.write_to_file(refine_structure, os.getcwd() + '/experiments/{}_{}_{}/'.format(_id, source, target) + 'source_tree.txt')

    # Create word embeddings and calculate similarities
    targets = [t.replace('.', '').replace('+', '').replace('-', '') for t in set(bk[target]) if t.split('(')[0] != to_predicate]
    similarities = transfer.similarity_fasttext(preds_learned, targets, fastTextModel, method=params.METHOD)
    #similarities = transfer.similarity_word2vec(preds_learned, targets, word2vecModel, method=params.METHOD)
    
    # Map source predicates to targets and creates transfer file
    mapping = transfer.map_predicates(preds_learned, similarities)
    transfer.write_to_file_closest_distance(predicate, to_predicate, arity, mapping, 'experiments/' + experiment_title, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)

    # Load predicate target dataset
    tar_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=params.SEED)

    # Set number of folds
    if target in ['nell_sports', 'nell_finances', 'yago2s']:
        n_folds = params.N_FOLDS
    else:
        n_folds = len(tar_data[0])

    # Dataframes to keep folds and confusion matrix values
    all_folds_results = pd.DataFrame([], columns=['CLL', 'AUC ROC', 'AUC PR', 'Total Learning Time', 'Total Inference Time'])
    confusion_matrix  = pd.DataFrame([], columns=['TP', 'FP', 'TN', 'FN'])
    
    target_trees = []
    for i in range(n_folds):
        logging.info('Starting fold {} \n'.format(str(i+1)))
    
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
        
        logging.info('Start transfer learning experiment\n')

        logging.info('Target train facts examples: %s' % len(tar_train_facts))
        logging.info('Target train pos examples: %s' % len(tar_train_pos))
        logging.info('Target train neg examples: %s\n' % len(tar_train_neg))
        logging.info('Target test facts examples: %s' % len(tar_test_facts))
        logging.info('Target test pos examples: %s' % len(tar_test_pos))
        logging.info('Target test neg examples: %s\n' % len(tar_test_neg))

        start = time.time()

        # Train model using transfer learning
        background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)
        model = boostsrl.train(background, tar_train_pos, tar_train_neg, tar_train_facts, refine=params.REFINE_FILENAME, transfer=params.TRANSFER_FILENAME, trees=params.TREES)
        
        end = time.time()
        learning_time = end-start

        logging.info('Model training time using transfer learning {}'.format(learning_time))

        start = time.time()

        # Test transfered model
        results = boostsrl.test(model, tar_test_pos, tar_test_neg, tar_test_facts, trees=params.TREES)

        end = time.time()
        inference_time = end-start

        # Get testing results

        #inference_time = results.testtime()
        t_results = results.summarize_results()
        results = {}
        #t_results['Learning time'] = learning_time
        #t_results['Inference time'] = inference_time
        results['CLL']     = t_results['CLL']
        results['AUC ROC'] = t_results['AUC ROC']
        results['AUC PR']  = t_results['AUC PR']
        #results.append('Precision: {}'.format(t_results['Precision'][0]))
        #results.append('Recall: {}'.format(t_results['Recall']))
        #results.append('F1: {}'.format(t_results['F1']))
        #results.append('\n')
        results['Total Learning Time']  = learning_time
        results['Total Inference Time'] = inference_time

        all_folds_results = all_folds_results.append(results, ignore_index=True)

        # Get target trees
        structured = []
        for i in range(params.TREES):
          structured.append(model.get_structured_tree(treenumber=i+1).copy())

        refine_structure = utils.get_all_rules_from_tree(structured)
        target_trees += refine_structure

        # Get confusion matrix by reading results from db files created by the Java application
        logging.info('Converting results file to txt')

        utils.convert_db_to_txt(to_predicate, params.TEST_OUTPUT)
        y_true, y_pred = utils.read_results(params.TEST_OUTPUT.format(to_predicate).replace('.db', '.txt'))

        logging.info('Building confusion matrix')

        # True Negatives, False Positives, False Negatives, True Positives
        TN, FP, FN, TP = utils.get_confusion_matrix(y_true, y_pred)

        confusion_matrix = confusion_matrix.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}, ignore_index=True)

    # Save all CV results
    all_folds_results.to_csv(os.getcwd() + '/experiments/{}_{}_{}/all_folds.txt'.format(_id, source, target))
    confusion_matrix.to_csv(os.getcwd() + '/experiments/{}_{}_{}/confusion_matrix.txt'.format(_id, source, target), index=False)
    utils.write_to_file(target_trees, os.getcwd() + '/experiments/{}_{}_{}/results.txt'.format(_id, source, target))