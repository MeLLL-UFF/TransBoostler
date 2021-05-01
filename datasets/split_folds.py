
from get_datasets import *
import numpy as np
import os

import sys
sys.path.append("..")

from experiments import experiments, bk
import parameters as params
import utils as utils

import logging

#verbose=True
source_balanced = 1
balanced = 1

for experiment in experiments:


    experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
    logging.info('Experiment {} \n'.format(experiment_title))

    _id = experiment['id']
    source = experiment['source']
    target = experiment['target']
    predicate = experiment['predicate']
    to_predicate = experiment['to_predicate']
    arity = experiment['arity']

    if os.path.exists(os.getcwd() + '/folds/{}'.format(target)):
        continue
    os.makedirs(os.getcwd() + '/folds/{}'.format(target))
    
    # Load predicate target dataset
    tar_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=params.SEED)

    # Set number of folds
    if target in ['nell_sports', 'nell_finances', 'yago2s', 'yeast2', 'fly']:
        n_folds = params.N_FOLDS
    else:
        n_folds = len(tar_data[0])

    for i in range(n_folds):
    
        # Group and shuffle
        if target not in ['nell_sports', 'nell_finances', 'yago2s', 'yeast2', 'fly']:
            [tar_train_facts, tar_test_facts] =  datasets.get_kfold_small(i, tar_data[0])
            [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, tar_data[1])
            [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, tar_data[2])
        else:
            [tar_train_facts, tar_test_facts] =  [tar_data[0][0], tar_data[0][0]]
            to_folds_pos = datasets.split_into_folds(tar_data[1][0], n_folds=n_folds, seed=params.SEED)
            to_folds_neg = datasets.split_into_folds(tar_data[2][0], n_folds=n_folds, seed=params.SEED)
            [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, to_folds_pos)
            [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, to_folds_neg)
        
        logging.info('Target train facts examples: %s' % len(tar_train_facts))
        logging.info('Target train pos examples: %s' % len(tar_train_pos))
        logging.info('Target train neg examples: %s\n' % len(tar_train_neg))
        logging.info('Target test facts examples: %s' % len(tar_test_facts))
        logging.info('Target test pos examples: %s' % len(tar_test_pos))
        logging.info('Target test neg examples: %s\n' % len(tar_test_neg))

        os.makedirs(os.getcwd() + '/folds/{}/fold_{}'.format(target,i+1))

        # Write train folds into txt
        utils.write_to_file(tar_train_facts, os.getcwd() + '/folds/{}/fold_{}/train_facts.txt'.format(target,i+1))
        utils.write_to_file(tar_train_pos, os.getcwd() + '/folds/{}/fold_{}/train_pos.txt'.format(target,i+1))
        utils.write_to_file(tar_train_neg, os.getcwd() + '/folds/{}/fold_{}/train_neg.txt'.format(target,i+1))

        # Write test folds into txt
        utils.write_to_file(tar_test_facts, os.getcwd() + '/folds/{}/fold_{}/test_facts.txt'.format(target,i+1))
        utils.write_to_file(tar_test_pos, os.getcwd() + '/folds/{}/fold_{}/test_pos.txt'.format(target,i+1))
        utils.write_to_file(tar_test_neg, os.getcwd() + '/folds/{}/fold_{}/test_neg.txt'.format(target,i+1))