
#import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('app.log','w'),logging.StreamHandler()])

from experiments import experiments, bk, setups
from gensim.test.utils import datapath
from datasets.get_datasets import *
from boostsrl import boostsrl
import gensim.downloader as api
import parameters as params
import utils as utils
import numpy as np
import random
import json
import copy
import time
import sys
import os

#verbose=True
source_balanced = False
balanced = False

runTransBoostler = True
runRDNB = False
learn_from_source = True

experiment_title = ''

def save_experiment(data, experiment_title):
    if not os.path.exists('experiments/' + experiment_title):
        os.makedirs('experiments/' + experiment_title)
    results = []
    if os.path.isfile('experiments/rdnb.json'):
        with open('experiments/{}/rdnb.json'.format(experiment_title), 'r') as fp:
            results = json.load(fp)
    results.append(data)
    with open('experiments/{}/rdnb.json'.format(experiment_title), 'w') as fp:
        json.dump(results, fp)


def train_and_test(background, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts):
    '''
        Train RDN-B using transfer learning
    '''

    start = time.time()
    model = boostsrl.train(background, train_pos, train_neg, train_facts, refine=None, transfer=None, trees=params.TREES)
    
    end = time.time()
    learning_time = end-start

    utils.print_function('Model training time {}'.format(learning_time), experiment_title)

    will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(10)]
    for w in will:
        utils.print_function(w, experiment_title)

    start = time.time()

    # Test transfered model
    results = boostsrl.test(model, test_pos, test_neg, test_facts, trees=params.TREES)

    end = time.time()
    inference_time = end-start

    utils.print_function('Inference time using transfer learning {}'.format(inference_time), experiment_title)

    return model, results.summarize_results(), learning_time, inference_time

def get_confusion_matrix(to_predicate):
    # Get confusion matrix by reading results from db files created by the Java application
    utils.print_function('Converting results file to txt', experiment_title)

    utils.convert_db_to_txt(to_predicate, params.TEST_OUTPUT)
    y_true, y_pred = utils.read_results(params.TEST_OUTPUT.format(to_predicate).replace('.db', '.txt'))
    

    utils.print_function('Building confusion matrix', experiment_title)

    # True Negatives, False Positives, False Negatives, True Positives
    TN, FP, FN, TP = utils.get_confusion_matrix(y_true, y_pred)

    utils.print_function('Confusion matrix \n', experiment_title)
    matrix = ['TP: {}'.format(TP), 'FP: {}'.format(FP), 'TN: {}'.format(TN), 'FN: {}'.format(FN)]
    for m in matrix:
        utils.print_function(m, experiment_title)

    return {'TP': TP, 'FP': FP, 'TN':TN, 'FN': FN}

def clean_previous_experiments_stuff():
    utils.print_function('Cleaning previous experiment\'s mess', experiment_title)
    utils.delete_file(params.TRANSFER_FILENAME)
    utils.delete_file(params.REFINE_FILENAME)
    utils.delete_folder(params.TRAIN_FOLDER)
    utils.delete_folder(params.TEST_FOLDER)
    utils.delete_folder(params.BEST_MODEL_FOLDER)

def main():

    # Dictionaries to keep all experiments results
    #transboostler_experiments = {}
    rdnb_confusion_matrix = {}

    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    results = {}

    for experiment in experiments:

        experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
        
        target = experiment['target']

        # Load total target dataset
        tar_total_data = datasets.load(target, bk[target], seed=params.SEED)

        if target in ['nell_sports', 'nell_finances', 'yago2s']:
            n_runs = params.N_FOLDS
        else:
            n_runs = len(tar_total_data[0])

        results = { 'save': { }}

        if 'nodes' in locals():
            nodes.clear()

        utils.print_function('Starting experiment {} \n'.format(experiment_title), experiment_title)

        _id = experiment['id']
        source = experiment['source']
        target = experiment['target']
        predicate = experiment['predicate']
        to_predicate = experiment['to_predicate']
        arity = experiment['arity']
        
        if target in ['twitter', 'yeast']:
            recursion = True
        else:
            recursion = False

        path = os.getcwd() + '/experiments/' + experiment_title
        if not os.path.exists(path):
            os.mkdir(path)

        # Get targets
        sources = [s.replace('.', '').replace('+', '').replace('-', '') for s in set(bk[source]) if s.split('(')[0] != to_predicate and 'recursion_' not in s]
        targets = [t.replace('.', '').replace('+', '').replace('-', '') for t in set(bk[target]) if t.split('(')[0] != to_predicate and 'recursion_' not in t]

        results['save'] = {
        'experiment': 0,
        'n_runs': 0,
        'seed': 441773,
        'source_balanced' : False,
        'balanced' : False,
        'folds' : n_runs,
        'nodeSize' : params.NODESIZE,
        'numOfClauses' : params.NUMOFCLAUSES,
        'maxTreeDepth' : params.MAXTREEDEPTH
        }
        
        if('rdn-b' not in rdnb_confusion_matrix):
            rdnb_confusion_matrix['rdn-b'] = {}

        rdnb_confusion_matrix['rdn-b'] = []
        confusion_matrix = {key: {'TP': [], 'FP': [], 'TN': [], 'FN': []} for key in params.AMOUNTS} 

        if target in ['nell_sports', 'nell_finances', 'yago2s']:
            n_folds = params.N_FOLDS
        else:
            n_folds = len(tar_total_data[0])

        results_save, confusion_matrix_save = [], []
        for i in range(n_folds):
            utils.print_function('\n Starting fold {} of {} folds \n'.format(i+1, n_folds), experiment_title)

            ob_save, cm_save = {}, {}

            if target not in ['nell_sports', 'nell_finances', 'yago2s']:
                [tar_train_pos, tar_test_pos] = datasets.get_kfold(i, tar_total_data[0])
            else:
                t_total_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=params.SEED)
                tar_train_pos = datasets.split_into_folds(t_total_data[1][0], n_folds=n_folds, seed=params.SEED)[i] + t_total_data[0][0]

            # Load new predicate target dataset
            tar_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=params.SEED)

            # Group and shuffle
            if target not in ['nell_sports', 'nell_finances', 'yago2s']:
                [tar_train_facts, tar_test_facts] =  datasets.get_kfold(i, tar_data[0])
                [tar_train_pos, tar_test_pos] =  datasets.get_kfold(i, tar_data[1])
                [tar_train_neg, tar_test_neg] =  datasets.get_kfold(i, tar_data[2])
            else:
                [tar_train_facts, tar_test_facts] =  [tar_data[0][0], tar_data[0][0]]
                to_folds_pos = datasets.split_into_folds(tar_data[1][0], n_folds=n_folds, seed=params.SEED)
                to_folds_neg = datasets.split_into_folds(tar_data[2][0], n_folds=n_folds, seed=params.SEED)
                [tar_train_pos, tar_test_pos] =  datasets.get_kfold(i, to_folds_pos)
                [tar_train_neg, tar_test_neg] =  datasets.get_kfold(i, to_folds_neg)
            
            random.shuffle(tar_train_pos)
            random.shuffle(tar_train_neg)
            
            utils.print_function('Start transfer learning experiment\n', experiment_title)

            utils.print_function('Target train facts examples: %s' % len(tar_train_facts), experiment_title)
            utils.print_function('Target train pos examples: %s' % len(tar_train_pos), experiment_title)
            utils.print_function('Target train neg examples: %s\n' % len(tar_train_neg), experiment_title)
            utils.print_function('Target test facts examples: %s' % len(tar_test_facts), experiment_title)
            utils.print_function('Target test pos examples: %s' % len(tar_test_pos), experiment_title)
            utils.print_function('Target test neg examples: %s\n' % len(tar_test_neg), experiment_title)

            # Creating background
            background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)
            
            for amount in params.AMOUNTS:
                utils.print_function('Amount of data: ' + str(amount), experiment_title)
                part_tar_train_pos = tar_train_pos[:int(amount * len(tar_train_pos))]
                part_tar_train_neg = tar_train_neg[:int(amount * len(tar_train_neg))]

                # Train and test
                utils.print_function('Training from scratch \n', experiment_title)

                # Learn and test 
                model, t_results, learning_time, inference_time = train_and_test(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts)
                del model

                t_results['Learning time'] = learning_time
                ob_save['amount_' + str(amount)] = t_results
                
                utils.show_results(utils.get_results_dict(t_results, learning_time, inference_time), experiment_title)

                #experiment_metrics[amount]['CLL'].append(t_results['CLL'])
                #experiment_metrics[amount]['AUC ROC'].append(t_results['AUC ROC'])
                #experiment_metrics[amount]['AUC PR'].append(t_results['AUC PR'])
                #experiment_metrics[amount]['Learning Time'].append(learning_time)
                #experiment_metrics[amount]['Inference Time'].append(inference_time)

                #transboostler_experiments[embeddingModel][similarityMetric].append(experiment_metrics)

                cm = get_confusion_matrix(to_predicate)
                cm_save['rdn-b'] = cm

                confusion_matrix[amount]['TP'].append(cm['TP'])
                confusion_matrix[amount]['FP'].append(cm['FP'])
                confusion_matrix[amount]['TN'].append(cm['TN'])
                confusion_matrix[amount]['FN'].append(cm['FN'])

                rdnb_confusion_matrix['rdn-b'].append(confusion_matrix) 
                del cm, t_results, learning_time, inference_time

            results_save.append(ob_save)
        save_experiment(results_save, experiment_title) 
    
        matrix_filename = os.getcwd() + '/experiments/{}_{}_{}/rdnb_confusion_matrix.json'.format(_id, source, target)
        #folds_filename  = os.getcwd() + '/experiments/{}_{}_{}/transboostler_curves_folds.json'.format(_id, source, target)

        # Save all results using transfer
        utils.save_json_file(matrix_filename, rdnb_confusion_matrix)
        #utils.save_json_file(folds_filename, transboostler_experiments)         

if __name__ == '__main__':
    sys.exit(main())
