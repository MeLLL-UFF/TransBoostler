
#import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('app.log','w'),logging.StreamHandler()])

from experiments import experiments, bk, setups
from gensim.test.utils import datapath
from datasets.get_datasets import *
from boostsrl import boostsrl
import parameters as params
import utils as utils
import numpy as np
import random
import time
import sys
import os

#verbose=True
source_balanced = False
balanced = False

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
        Train RDN-B
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

    # Test model
    results = boostsrl.test(model, test_pos, test_neg, test_facts, trees=params.TREES)

    end = time.time()
    inference_time = end-start

    utils.print_function('Inference time {}'.format(inference_time), experiment_title)

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

    # Converts to int to fix JSON np.int64 problem
    return {'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN)}

def main():

    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    results, confusion_matrix = {}, {}

    # Dictionaries to keep all experiments results
    #transboostler_experiments = {}
    rdnb_confusion_matrix = {}

    for experiment in experiments:

        confusion_matrix_save_all = []

        experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
        
        target = experiment['target']

        # Load total target dataset
        tar_total_data = datasets.load(target, bk[target], seed=params.SEED)

        if target in ['nell_sports', 'nell_finances', 'yago2s']:
            n_runs = params.N_FOLDS
        else:
            n_runs = len(tar_total_data[0])

        results = { 'save': { }}

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

        # Get sources and targets
        sources = [s.replace('.', '').replace('+', '').replace('-', '') for s in set(bk[source]) if s.split('(')[0] != to_predicate and 'recursion_' not in s]
        targets = [t.replace('.', '').replace('+', '').replace('-', '') for t in set(bk[target]) if t.split('(')[0] != to_predicate and 'recursion_' not in t]
        
        path = os.getcwd() + '/experiments/' + experiment_title
        if not os.path.exists(path):
            os.mkdir(path)

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
            #transboostler_experiments[embeddingModel] = {}
            rdnb_confusion_matrix['rdn-b'] = {}

        #transboostler_experiments[embeddingModel][similarityMetric] = []
        #experiment_metrics = {key: {'CLL': [], 'AUC ROC': [], 'AUC PR': [], 'Learning Time': [], 'Inference Time': []} for key in params.AMOUNTS} 
        rdnb_confusion_matrix['rdn-b'] = []
        confusion_matrix = {'TP': [], 'FP': [], 'TN': [], 'FN': []} 

        utils.print_function('Starting experiments for RDN-B \n', experiment_title)

        if target in ['nell_sports', 'nell_finances', 'yago2s']:
            n_folds = params.N_FOLDS
        else:
            n_folds = len(tar_total_data[0])

        results_save, confusion_matrix_save = [], []
        for i in range(n_folds):
            utils.print_function('\n Starting fold {} of {} folds \n'.format(i+1, n_folds), experiment_title)

            ob_save, cm_save = {}, {}

            if target not in ['nell_sports', 'nell_finances', 'yago2s']:
                [tar_train_pos, tar_test_pos] = datasets.get_kfold_small(i, tar_total_data[0])
            else:
                t_total_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=params.SEED)
                tar_train_pos = datasets.split_into_folds(t_total_data[1][0], n_folds=n_folds, seed=params.SEED)[i] + t_total_data[0][0]

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
            
            random.shuffle(tar_train_pos)
            random.shuffle(tar_train_neg)
            
            utils.print_function('Start training from scratch\n', experiment_title)

            utils.print_function('Target train facts examples: %s' % len(tar_train_facts), experiment_title)
            utils.print_function('Target train pos examples: %s' % len(tar_train_pos), experiment_title)
            utils.print_function('Target train neg examples: %s\n' % len(tar_train_neg), experiment_title)

            utils.print_function('Target test facts examples: %s' % len(tar_test_facts), experiment_title)
            utils.print_function('Target test pos examples: %s' % len(tar_test_pos), experiment_title)
            utils.print_function('Target test neg examples: %s\n' % len(tar_test_neg), experiment_title)

            # Creating background
            background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)

            # Train and test
            utils.print_function('Training from scratch \n', experiment_title)

            # Learn and test model not revising theory
            model, t_results, learning_time, inference_time = train_and_test(background, tar_train_pos, tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts)
            del model

            t_results['Learning time'] = learning_time
            ob_save['rdn-b'] = t_results
            
            utils.show_results(utils.get_results_dict(t_results, learning_time, inference_time), experiment_title)

            cm = get_confusion_matrix(to_predicate)
            cm_save['rdn-b'] = cm

            confusion_matrix['TP'].append(cm['TP'])
            confusion_matrix['FP'].append(cm['FP'])
            confusion_matrix['TN'].append(cm['TN'])
            confusion_matrix['FN'].append(cm['FN'])

            rdnb_confusion_matrix['rdn-b'].append(confusion_matrix) 
            del cm, t_results, learning_time, inference_time

            results_save.append(ob_save)
        save_experiment(results_save, experiment_title)
        
        matrix_filename = os.getcwd() + '/experiments/{}_{}_{}/rdnb_confusion_matrix.json'.format(_id, source, target)

        # Save all results
        utils.save_json_file(matrix_filename, rdnb_confusion_matrix)

if __name__ == '__main__':
    sys.exit(main())
