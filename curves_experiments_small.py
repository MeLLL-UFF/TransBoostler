
#import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('app.log','w'),logging.StreamHandler()])

from ekphrasis.classes.segmenter import Segmenter
from experiments import experiments, bk, setups
from gensim.models import KeyedVectors, FastText
from gensim.test.utils import datapath
from revision import TheoryRevision
from datasets.get_datasets import *
from similarity import Similarity
from boostsrl import boostsrl
from transfer import Transfer
import gensim.downloader as api
import parameters as params
import utils as utils
import numpy as np
import random
import pickle
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
learn_from_source = False

folds_path = 'folds_transfer_experiment'

revision = TheoryRevision()
segmenter = Segmenter(corpus="english")

experiment_title = ''

def save_experiment(data, experiment_title, embeddingModel, similarityMetric):
    if not os.path.exists('experiments/' + experiment_title):
        os.makedirs('experiments/' + experiment_title)
    results = []
    if os.path.isfile('experiments/' + experiment_title + '/' + experiment_title + '_{}_{}.json'.format(embeddingModel, similarityMetric)):
        with open('experiments/' + experiment_title + '/' + experiment_title + '_{}_{}.json'.format(embeddingModel, similarityMetric), 'r') as fp:
            results = json.load(fp)
    results.append(data)
    with open('experiments/' + experiment_title + '/' + experiment_title + '_{}_{}.json'.format(embeddingModel, similarityMetric), 'w') as fp:
        json.dump(results, fp)

def load_model(model_name):

  if(model_name == 'fasttext'):
    if not os.path.exists(params.WIKIPEDIA_FASTTEXT): 
        raise ValueError("SKIP: You need to download the fasttext wikipedia model")

    if 'loadedModel' not in locals():
        utils.print_function('Loading fasttext model', experiment_title)
        start = time.time()
        #loadedModel = FastText.load_fasttext_format(params.WIKIPEDIA_FASTTEXT)
        loadedModel = KeyedVectors.load_word2vec_format(params.WIKIPEDIA_FASTTEXT, binary=False, unicode_errors='ignore')

        end = time.time()
        utils.print_function('Time to load FastText model: {} seconds'.format(round(end-start, 2)), experiment_title)

  elif(model_name == 'word2vec'):

    if not os.path.exists(params.GOOGLE_WORD2VEC): 
        raise ValueError("SKIP: You need to download the google news model")

    if 'loadedModel' not in locals():
        utils.print_function('Loading word2vec model', experiment_title)
        start = time.time()
        loadedModel = KeyedVectors.load_word2vec_format(params.GOOGLE_WORD2VEC, binary=True, unicode_errors='ignore')

        end = time.time()
        utils.print_function('Time to load Word2Vec model: {} seconds'.format(round(end-start, 2)), experiment_title)
  else:
    raise ValueError("SKIP: Embedding models must be 'fasttext' or 'word2vec'")

  return loadedModel

def train_and_test(background, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, refine=None, transfer=None):
    '''
        Train RDN-B using transfer learning
    '''

    start = time.time()
    model = boostsrl.train(background, train_pos, train_neg, train_facts, refine=refine, transfer=transfer, trees=params.TREES)
    
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

def get_confusion_matrix(to_predicate, revision=False):
    # Get confusion matrix by reading results from db files created by the Java application
    utils.print_function('Converting results file to txt', experiment_title)

    if revision:
        utils.convert_db_to_txt(to_predicate, params.TEST_OUTPUT.replace('test', 'best/test'))
        y_true, y_pred = utils.read_results(params.TEST_OUTPUT.replace('test', 'best/test').format(to_predicate).replace('.db', '.txt'))
    else:
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

def save_pickle_file(nodes, _id, source, target, filename):
    with open(os.getcwd() + '/experiments/{}_{}_{}/{}'.format(_id, source, target, filename), 'wb') as file:
        pickle.dump(nodes, file)

def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():

    # Dictionaries to keep all experiments results
    #transboostler_experiments = {}
    transboostler_confusion_matrix = {}

    if not os.path.exists('experiments'):
        os.makedirs('experiments')
        os.makedirs('experiments/similarities')
        os.makedirs('experiments/similarities/fasttext')
        os.makedirs('experiments/similarities/fasttext/cosine')
        os.makedirs('experiments/similarities/fasttext/softcosine')
        os.makedirs('experiments/similarities/fasttext/euclidean')
        os.makedirs('experiments/similarities/fasttext/wmd')

    results = {}

    for experiment in experiments:

        experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
        
        target = experiment['target']
    
        # n_runs = n_files - path - 1
        n_runs = len(list(os.walk('datasets/folds/{}/'.format(target)))) - 1

        results = { 'save': { }}

        if 'nodes' in locals():
            nodes.clear()
        
        #Clean folders if exists
        clean_previous_experiments_stuff()

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
                
        if(learn_from_source):

            # Load source dataset
            src_total_data = datasets.load(source, bk[source], seed=params.SEED)
            src_data = datasets.load(source, bk[source], target=predicate, balanced=source_balanced, seed=params.SEED)

            # Group and shuffle
            src_facts = datasets.group_folds(src_data[0])
            src_pos   = datasets.group_folds(src_data[1])
            src_neg   = datasets.group_folds(src_data[2])
                        
            utils.print_function('Start learning from source dataset\n')
            
            utils.print_function('Source train facts examples: {}'.format(len(src_facts)), experiment_title)
            utils.print_function('Source train pos examples: {}'.format(len(src_pos)), experiment_title)
            utils.print_function('Source train neg examples: {}\n'.format(len(src_neg)), experiment_title)
            
            start = time.time()

            # Learning from source dataset
            background = boostsrl.modes(bk[source], [predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)
            model = boostsrl.train(background, src_pos, src_neg, src_facts, trees=params.TREES)
            
            end = time.time()

            #TODO: adicionar o tempo corretamente
            #print('Model training time {}'.format(model.traintime()))

            utils.print_function('Model training time {} \n'.format(end-start), experiment_title)

            utils.print_function('Building refine structure \n', experiment_title)

            # Get all learned trees
            structured = []
            for i in range(params.TREES):
                structured.append(model.get_structured_tree(treenumber=i+1).copy())
            
            will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(10)]
            for w in will:
                utils.print_function(w, experiment_title)
            
            del model

            # Get the list of predicates from source tree          
            nodes = utils.deep_first_search_nodes(structured, utils.match_bk_source(set(bk[source])))
            save_pickle_file(nodes, _id, source, target, params.SOURCE_TREE_NODES_FILES)
            save_pickle_file(structured, _id, source, target, params.STRUCTURED_TREE_NODES_FILES)

            # Get all rules learned by RDN-B
            refine_structure = utils.get_all_rules_from_tree(structured)
            utils.write_to_file(refine_structure, params.REFINE_FILENAME)
            utils.write_to_file(refine_structure, os.getcwd() + '/experiments/{}_{}_{}/{}'.format(_id, source, target, params.REFINE_FILENAME.split('/')[1]))

        else:
            utils.print_function('Loading pre-trained trees.', experiment_title)

            from shutil import copyfile
            copyfile(os.getcwd() + '/resources/{}_{}_{}/{}'.format(_id, source, target, params.REFINE_FILENAME.split('/')[1]), os.getcwd() + '/' + params.REFINE_FILENAME)
            nodes = load_pickle_file(os.getcwd() + '/resources/{}_{}_{}/{}'.format(_id, source, target, params.SOURCE_TREE_NODES_FILES))
            structured = load_pickle_file(os.getcwd() + '/resources/{}_{}_{}/{}'.format(_id, source, target, params.STRUCTURED_TREE_NODES_FILES))

        # Get targets
        sources = [s.replace('.', '').replace('+', '').replace('-', '') for s in set(bk[source]) if s.split('(')[0] != to_predicate and 'recursion_' not in s]
        targets = [t.replace('.', '').replace('+', '').replace('-', '') for t in set(bk[target]) if t.split('(')[0] != to_predicate and 'recursion_' not in t]

        for setup in setups: 
            embeddingModel = setup['model'].lower()
            similarityMetric = setup['similarity_metric'].lower()
            theoryRevision = setup['revision_theory']

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
            
            utils.delete_file(params.TRANSFER_FILENAME)
            
            if(embeddingModel not in transboostler_confusion_matrix):
            #    transboostler_experiments[embeddingModel] = {}
                transboostler_confusion_matrix[embeddingModel] = {}

            #transboostler_experiments[embeddingModel][similarityMetric] = []
            #experiment_metrics = {key: {'CLL': [], 'AUC ROC': [], 'AUC PR': [], 'Learning Time': [], 'Inference Time': []} for key in params.AMOUNTS} 
            transboostler_confusion_matrix[embeddingModel][similarityMetric] = []
            confusion_matrix = {key: {'TP': [], 'FP': [], 'TN': [], 'FN': []} for key in params.AMOUNTS} 

            utils.print_function('Starting experiments for {} using {} \n'.format(embeddingModel, similarityMetric), experiment_title)
        
            if('previous' not in locals() or previous != embeddingModel):
                loadedModel = load_model(embeddingModel)
                previous = embeddingModel

            transfer = Transfer(model=loadedModel, model_name=embeddingModel, segmenter=segmenter, similarity_metric=similarityMetric, sources=sources, targets=targets)

            while results['save']['n_runs'] < n_runs:
                utils.print_function('Run: ' + str(results['save']['n_runs'] + 1), experiment_title)
            
                start = time.time()

                # Map and transfer using the loaded embedding model
                mapping  = transfer.map_predicates(similarityMetric, nodes, targets)
                transfer.write_to_file_closest_distance(similarityMetric, embeddingModel, predicate, to_predicate, arity, mapping, 'experiments/' + experiment_title, recursion=recursion, searchArgPermutation=params.SEARCH_PERMUTATION, searchEmpty=params.SEARCH_EMPTY, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)
                #transfer.write_constraints_to_file('experiments/' + experiment_title)
                del mapping

                end = time.time()
                mapping_time = end-start

                # Load total target dataset
                tar_total_data = datasets.load(target, bk[target], seed=results['save']['seed'])

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
                    
                    for amount in params.AMOUNTS_SMALL:
                        utils.print_function('Amount of data: ' + str(amount), experiment_title)
                        part_tar_train_pos = tar_train_pos[:amount]
                        part_tar_train_neg = tar_train_neg[:amount]

                        # Train and test using transfer learning
                        utils.print_function('Training using transfer \n', experiment_title)

                        if(theoryRevision):
                            # Learn and test model applying revision theory
                            t_results, learning_time, inference_time = revision.apply(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, structured, experiment_title)
                        else:
                            # Learn and test model not revising theory
                            model, t_results, learning_time, inference_time = train_and_test(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, params.REFINE_FILENAME, params.TRANSFER_FILENAME)
                            del model

                        t_results['Learning time'] = learning_time + mapping_time
                        t_results['Mapping time'] = mapping_time
                        ob_save['transfer_' + str(amount)] = t_results
                        
                        learning_time += mapping_time
                        utils.show_results(utils.get_results_dict(t_results, learning_time, inference_time), experiment_title)

                        #experiment_metrics[amount]['CLL'].append(t_results['CLL'])
                        #experiment_metrics[amount]['AUC ROC'].append(t_results['AUC ROC'])
                        #experiment_metrics[amount]['AUC PR'].append(t_results['AUC PR'])
                        #experiment_metrics[amount]['Learning Time'].append(learning_time)
                        #experiment_metrics[amount]['Inference Time'].append(inference_time)

                        #transboostler_experiments[embeddingModel][similarityMetric].append(experiment_metrics)

                        cm = get_confusion_matrix(to_predicate, revision=theoryRevision)
                        cm_save['transfer'] = cm

                        confusion_matrix[amount]['TP'].append(cm['TP'])
                        confusion_matrix[amount]['FP'].append(cm['FP'])
                        confusion_matrix[amount]['TN'].append(cm['TN'])
                        confusion_matrix[amount]['FN'].append(cm['FN'])

                        transboostler_confusion_matrix[embeddingModel][similarityMetric].append(confusion_matrix) 
                        del cm, t_results, learning_time, inference_time
                
                        previous = setup['model'].lower()

                    results_save.append(ob_save)
                save_experiment(results_save, experiment_title, embeddingModel, similarityMetric)
                results['save']['n_runs'] += 1    
        
        matrix_filename = os.getcwd() + '/experiments/{}_{}_{}/transboostler_confusion_matrix.json'.format(_id, source, target)
        #folds_filename  = os.getcwd() + '/experiments/{}_{}_{}/transboostler_curves_folds.json'.format(_id, source, target)
        
        if(theoryRevision):
            matrix_filename = matrix_filename.replace('.json', '_revision.json')
            #folds_filename  = folds_filename.replace('.json', '_revision.json')

        # Save all results using transfer
        utils.save_json_file(matrix_filename, transboostler_confusion_matrix)
        #utils.save_json_file(folds_filename, transboostler_experiments)         

if __name__ == '__main__':
    sys.exit(main())
