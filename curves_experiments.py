
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('app.log','w'),logging.StreamHandler()])

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

revision = TheoryRevision()
segmenter = Segmenter(corpus="english")

def load_model(model_name):

  if(model_name == 'fasttext'):
    if not os.path.exists(params.WIKIPEDIA_FASTTEXT): 
        raise ValueError("SKIP: You need to download the fasttext wikipedia model")

    if 'loadedModel' not in locals():
        logging.info('Loading fasttext model')
        start = time.time()
        #loadedModel = FastText.load_fasttext_format(params.WIKIPEDIA_FASTTEXT)
        loadedModel = KeyedVectors.load_word2vec_format(params.WIKIPEDIA_FASTTEXT, binary=False, unicode_errors='ignore')

        end = time.time()
        logging.info('Time to load FastText model: {} seconds'.format(round(end-start, 2)))

  elif(model_name == 'word2vec'):

    if not os.path.exists(params.GOOGLE_WORD2VEC): 
        raise ValueError("SKIP: You need to download the google news model")

    if 'loadedModel' not in locals():
        logging.info('Loading word2vec model')
        start = time.time()
        loadedModel = KeyedVectors.load_word2vec_format(params.GOOGLE_WORD2VEC, binary=True, unicode_errors='ignore')

        end = time.time()
        logging.info('Time to load Word2Vec model: {} seconds'.format(round(end-start, 2)))
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

    logging.info('Model training time {}'.format(learning_time))

    will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(10)]
    for w in will:
        logging.info(w)

    start = time.time()

    # Test transfered model
    results = boostsrl.test(model, test_pos, test_neg, test_facts, trees=params.TREES)

    end = time.time()
    inference_time = end-start

    logging.info('Inference time using transfer learning {}'.format(inference_time))

    return model, results.summarize_results(), learning_time, inference_time

def get_confusion_matrix(to_predicate, revision=False):
    # Get confusion matrix by reading results from db files created by the Java application
    logging.info('Converting results file to txt')

    if revision:
        utils.convert_db_to_txt(to_predicate, params.TEST_OUTPUT.replace('test', 'best/test'))
        y_true, y_pred = utils.read_results(params.TEST_OUTPUT.replace('test', 'best/test').format(to_predicate).replace('.db', '.txt'))
    else:
        utils.convert_db_to_txt(to_predicate, params.TEST_OUTPUT)
        y_true, y_pred = utils.read_results(params.TEST_OUTPUT.format(to_predicate).replace('.db', '.txt'))
    

    logging.info('Building confusion matrix')

    # True Negatives, False Positives, False Negatives, True Positives
    TN, FP, FN, TP = utils.get_confusion_matrix(y_true, y_pred)

    logging.info('Confusion matrix \n')
    matrix = ['TP: {}'.format(TP), 'FP: {}'.format(FP), 'TN: {}'.format(TN), 'FN: {}'.format(FN)]
    for m in matrix:
        logging.info(m)

    return {'TP': TP, 'FP': FP, 'TN':TN, 'FN': FN}

def clean_previous_experiments_stuff():
    logging.info('Cleaning previous experiment\'s mess')
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
    transboostler_experiments = {}
    transboostler_confusion_matrix = {}

    if not os.path.exists('experiments'):
        os.makedirs('experiments')
        os.makedirs('experiments/similarities')
        os.makedirs('experiments/similarities/Fasttext')
        
    for experiment in experiments:

        if 'nodes' in locals():
            nodes.clear()
        
        #Clean folders if exists
        clean_previous_experiments_stuff()

        experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
        logging.info('Starting experiment {} \n'.format(experiment_title))

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

        if(runTransBoostler):
            
            if(learn_from_source):

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
                background = boostsrl.modes(bk[source], [predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)
                model = boostsrl.train(background, src_pos, src_neg, src_facts, trees=params.TREES)
                
                end = time.time()

                #TODO: adicionar o tempo corretamente
                #print('Model training time {}'.format(model.traintime()))

                logging.info('Model training time {} \n'.format(end-start))

                logging.info('Building refine structure \n')

                # Get all learned trees
                structured = []
                for i in range(params.TREES):
                    structured.append(model.get_structured_tree(treenumber=i+1).copy())
                
                will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(10)]
                for w in will:
                    logging.info(w)
                
                del model

                # Get the list of predicates from source tree          
                nodes = utils.deep_first_search_nodes(structured, utils.match_bk_source(set(bk[source])))
                save_pickle_file(nodes, _id, source, target, params.SOURCE_TREE_NODES_FILES)
                save_pickle_file(structured, _id, source, target, params.STRUCTURED_TREE_NODES_FILES)

                # Get all rules learned by RDN-B
                refine_structure = utils.get_all_rules_from_tree(structured)
                utils.write_to_file(refine_structure, params.REFINE_FILENAME)
                utils.write_to_file(refine_structure, os.getcwd() + '/experiments/{}_{}_{}/{}'.format(_id, source, target, params.REFINE_FILENAME.split('/')[1]))

            if(not learn_from_source):
                logging.info('Loading pre-trained trees.')

                from shutil import copyfile
                copyfile(os.getcwd() + '/experiments/{}_{}_{}/{}'.format(_id, source, target, params.REFINE_FILENAME.split('/')[1]), os.getcwd() + '/' + params.REFINE_FILENAME)
                nodes = load_pickle_file(os.getcwd() + '/experiments/{}_{}_{}/{}'.format(_id, source, target, params.SOURCE_TREE_NODES_FILES))
                #structured = load_pickle_file(os.getcwd() + '/experiments/{}_{}_{}/{}'.format(_id, source, target, params.STRUCTURED_TREE_NODES_FILES))

            # Get targets
            targets = [t.replace('.', '').replace('+', '').replace('-', '') for t in set(bk[target]) if t.split('(')[0] != to_predicate and 'recursion_' not in t]

            for setup in setups: 
                embeddingModel = setup['model'].lower()
                similarityMetric = setup['similarity_metric'].lower()
                theoryRevision = setup['revision_theory']
                
                utils.delete_file(params.TRANSFER_FILENAME)
                
                if(embeddingModel not in transboostler_experiments):
                    transboostler_experiments[embeddingModel] = {}
                    transboostler_confusion_matrix[embeddingModel] = {}

                transboostler_experiments[embeddingModel][similarityMetric] = {key: {'CLL': [], 'AUC ROC': [], 'AUC PR': [], 'Learning Time': [], 'Inference Time': []} for key in params.AMOUNTS} 
                transboostler_confusion_matrix[embeddingModel][similarityMetric] = {key: {'TP': [], 'FP': [], 'TN': [], 'FN': []} for key in params.AMOUNTS} 

                logging.info('Starting experiments for {} using {} \n'.format(embeddingModel, similarityMetric))
            
                if('previous' not in locals() or previous != embeddingModel):
                    loadedModel = load_model(embeddingModel)
                    previous = embeddingModel
                
                transfer = Transfer(model=loadedModel, model_name=embeddingModel, segmenter=segmenter)
                
                start = time.time()

                # Map and transfer using the loaded embedding model
                mapping  = transfer.map_predicates(similarityMetric, nodes, targets)
                transfer.write_to_file_closest_distance(similarityMetric, embeddingModel, predicate, to_predicate, arity, mapping, 'experiments/' + experiment_title, recursion=recursion, searchArgPermutation=params.SEARCH_PERMUTATION, searchEmpty=params.SEARCH_EMPTY, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)
                #transfer.write_constraints_to_file('experiments/' + experiment_title)
                del mapping

                end = time.time()
                mapping_time = end-start

                # Set number of folds
                n_folds = datasets.get_n_folds(target)

                for i in range(n_folds):
                    logging.info('\n Starting fold {} of {} folds \n'.format(i+1, n_folds))

                    [tar_train_facts, tar_test_facts] =  datasets.load_pre_saved_folds(i+1, target, 'facts')
                    [tar_train_pos, tar_test_pos]     =  datasets.load_pre_saved_folds(i+1, target, 'pos')
                    [tar_train_neg, tar_test_neg]     =  datasets.load_pre_saved_folds(i+1, target, 'neg')
                    
                    random.shuffle(tar_train_pos)
                    random.shuffle(tar_train_neg)
                    
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

                        # Creating background
                        background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)

                        # Get testing results
                        logging.info('Training using transfer \n')

                        if(theoryRevision):
                            # Learn and test model applying revision theory
                            t_results, learning_time, inference_time = revision.apply(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, structured)
                        else:
                            # Learn and test model not revising theory
                            model, t_results, learning_time, inference_time = train_and_test(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, params.REFINE_FILENAME, params.TRANSFER_FILENAME)
                            del model
                        
                        learning_time += mapping_time
                        utils.show_results(utils.get_results_dict(t_results, learning_time, inference_time))

                        transboostler_experiments[embeddingModel][similarityMetric][amount]['CLL'].append(t_results['CLL'])
                        transboostler_experiments[embeddingModel][similarityMetric][amount]['AUC ROC'].append(t_results['AUC ROC'])
                        transboostler_experiments[embeddingModel][similarityMetric][amount]['AUC PR'].append(t_results['AUC PR'])
                        transboostler_experiments[embeddingModel][similarityMetric][amount]['Learning Time'].append(learning_time)
                        transboostler_experiments[embeddingModel][similarityMetric][amount]['Inference Time'].append(inference_time)

                        cm = get_confusion_matrix(to_predicate, revision=theoryRevision)
                        transboostler_confusion_matrix[embeddingModel][similarityMetric][amount]['TP'].append(cm['TP']) 
                        transboostler_confusion_matrix[embeddingModel][similarityMetric][amount]['FP'].append(cm['FP']) 
                        transboostler_confusion_matrix[embeddingModel][similarityMetric][amount]['TN'].append(cm['TN']) 
                        transboostler_confusion_matrix[embeddingModel][similarityMetric][amount]['FN'].append(cm['FN']) 
                        del cm, t_results, learning_time, inference_time
                        previous = setup['model'].lower()

        if(runRDNB):
            # Dictionary to keep amounts values
            RDNB_results = {key: {'CLL': [], 'AUC ROC': [], 'AUC PR': [], 'Learning Time': [], 'Inference Time': []} for key in params.AMOUNTS}
            RDNB_confusion_matrix = {key: {'TP': [], 'FP': [], 'TN': [], 'FN': []} for key in params.AMOUNTS}
            
            # Set number of folds
            n_folds = datasets.get_n_folds(target)

            for i in range(n_folds):
                logging.info('\n Starting fold {} of {} folds \n'.format(i+1, n_folds))

                [tar_train_facts, tar_test_facts] =  datasets.load_pre_saved_folds(i+1, target, 'facts')
                [tar_train_pos, tar_test_pos]     =  datasets.load_pre_saved_folds(i+1, target, 'pos')
                [tar_train_neg, tar_test_neg]     =  datasets.load_pre_saved_folds(i+1, target, 'neg')
                
                logging.info('Starting to learn from scratch\n')

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

                    # Creating background
                    background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)

                    # Get testing results
                    model, t_results, learning_time, inference_time = train_and_test(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_train_pos, tar_test_neg, tar_test_facts)

                    # Get target trees
                    structured = []
                    for i in range(params.TREES):
                        structured.append(model.get_structured_tree(treenumber=i+1).copy())

                    RDNB_results[amount]['CLL'].append(t_results['CLL'])
                    RDNB_results[amount]['AUC ROC'].append(t_results['AUC ROC'])
                    RDNB_results[amount]['AUC PR'].append(t_results['AUC PR'])
                    RDNB_results[amount]['Learning Time'].append(learning_time)
                    RDNB_results[amount]['Learning Time'].append(inference_time)

                    cm = get_confusion_matrix(to_predicate)
                    RDNB_confusion_matrix[amount]['TP'].append(cm['TP'])
                    RDNB_confusion_matrix[amount]['FP'].append(cm['FP'])
                    RDNB_confusion_matrix[amount]['TN'].append(cm['TN'])
                    RDNB_confusion_matrix[amount]['FN'].append(cm['FN'])

                    results = {}
                    results = utils.get_results_dict(t_results, learning_time, inference_time)
                    utils.show_results(results)
                    del model, cm, t_results, learning_time, inference_time, results
                    

        if(runTransBoostler):
            
            matrix_filename = os.getcwd() + '/experiments/{}_{}_{}/transboostler_confusion_matrix.json'.format(_id, source, target)
            folds_filename  = os.getcwd() + '/experiments/{}_{}_{}/transboostler_curves_folds.json'.format(_id, source, target)
            
            if(theoryRevision):
                matrix_filename = matrix_filename.replace('.json', '_revision.json')
                folds_filename  = folds_filename.replace('.json', '_revision.json')

            # Save all results using transfer
            utils.save_json_file(matrix_filename, transboostler_confusion_matrix)
            utils.save_json_file(folds_filename, transboostler_experiments)         

        if(runRDNB):
            
            matrix_filename = os.getcwd() + '/experiments/{}_{}_{}/rdnb_curves_folds.json'.format(_id, source, target)
            folds_filename  = os.getcwd() + '/experiments/{}_{}_{}/rdnb_confusion_matrix.json'.format(_id, source, target)
            
            # Save all CV results
            utils.save_json_file(matrix_filename, RDNB_results)
            utils.save_json_file(folds_filename, RDNB_confusion_matrix)

if __name__ == '__main__':
    sys.exit(main())
