from experiments import experiments, bk
from datasets.get_datasets import *
from boostsrl import boostsrl
import parameters as params
import utils as utils
import numpy as np
import random
import time
import sys
import os

import logging
logging.basicConfig(level=logging.INFO)

#verbose=True
source_balanced = 1
balanced = 1

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

    # Load predicate target dataset
    tar_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=params.SEED)

    # Set number of folds
    if target in ['nell_sports', 'nell_finances', 'yago2s']:
        n_folds = params.N_FOLDS
    else:
        n_folds = len(tar_data[0])

    # Dictionary to keep amounts values
    RDNB_results   = {key: {'AUC ROC': 0, 'AUC PR': 0} for key in params.AMOUNTS}

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

        for amount in params.AMOUNTS:
            logging.info('Amount of data: ' + str(amount))
            part_tar_train_pos = tar_train_pos[:int(amount * len(tar_train_pos))]
            part_tar_train_neg = tar_train_neg[:int(amount * len(tar_train_neg))]

            # Train model from scratch
            background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)
            
            start = time.time()
            model = boostsrl.train(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, trees=params.TREES)
            end = time.time()
            learning_time = end-start

            will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(10)]
            utils.write_to_file(will, 'RDNB_{}.txt'.format(experiment_title))

            # Test transfered model
            start = time.time()
            results = boostsrl.test(model, tar_test_pos, tar_test_neg, tar_test_facts, trees=params.TREES)
            end = time.time()
            inference_time = end-start

            # Get testing results
            t_results = results.summarize_results()
            RDNB_results[amount]['CLL'] += t_results['CLL']
            RDNB_results[amount]['AUC ROC'] += t_results['AUC ROC']
            RDNB_results[amount]['AUC PR']  += t_results['AUC PR']
            RDNB_results[amount]['Precision']  += t_results['Precision']
            RDNB_results[amount]['Recall']  += t_results['Recall']
            RDNB_results[amount]['Total Learning Time']  += learning_time
            RDNB_results[amount]['Total Inference Time'] += inference_time

            if(amount == 1.0):

                results = {}
                results['CLL']     = t_results['CLL']
                results['AUC ROC'] = t_results['AUC ROC']
                results['AUC PR']  = t_results['AUC PR']
                results['Precision']  = t_results['Precision']
                results['Recall']  = t_results['Recall']
                results['Total Learning Time']  = learning_time
                results['Total Inference Time'] = inference_time
                
                all_folds_results = all_folds_results.append(results, ignore_index=True)

                utils.convert_db_to_txt(to_predicate, params.TEST_OUTPUT)
                y_true, y_pred = utils.read_results(params.TEST_OUTPUT.format(to_predicate).replace('.db', '.txt'))

                logging.info('Building confusion matrix')

                # True Negatives, False Positives, False Negatives, True Positives
                TN, FP, FN, TP = utils.get_confusion_matrix(y_true, y_pred)

                confusion_matrix = confusion_matrix.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}, ignore_index=True)

    all_RDNB_results = pd.DataFrame.from_dict(RDNB_results)
    all_RDNB_results = all_RDNB_results/n_folds
    
    all_RDNB_results.to_csv(os.getcwd() + '/experiments/{}_{}_{}/RDNB_curves.csv'.format(_id, source, target))

    # Save all CV results
    all_folds_results.to_csv(os.getcwd() + '/experiments/{}_{}_{}/RDNB_all_folds.txt'.format(_id, source, target))
    confusion_matrix.to_csv(os.getcwd() + '/experiments/{}_{}_{}/RDNB_confusion_matrix.txt'.format(_id, source, target), index=False)