
from ekphrasis.classes.segmenter import Segmenter
from gensim.models import KeyedVectors, FastText
from gensim.test.utils import get_tmpfile
from experiments import experiments, bk
from revision import TheoryRevision
from datasets.get_datasets import *
from similarity import Similarity
from boostsrl import boostsrl
from transfer import Transfer
import gensim.downloader as api
import parameters as params
import utils as utils
import numpy as np
import fasttext
import random
import copy
import time
import sys
import os

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler("app.log"),logging.StreamHandler()])

if not os.path.exists(params.WIKIPEDIA_FASTTEXT):
    raise ValueError("SKIP: You need to download the fasttext wikipedia model")

logging.info('Loading fasttext model')
start = time.time()

fastTextModel = fasttext.load_model(params.WIKIPEDIA_FASTTEXT)

end = time.time()
logging.info('Time to load FastText model: {} seconds'.format(round(end-start, 2)))

if not os.path.exists(params.GOOGLE_WORD2VEC):
    raise ValueError("SKIP: You need to download the google news model")

logging.info('Loading word2vec model')
start - time.time()

word2vecModel = KeyedVectors.load_word2vec_format(params.GOOGLE_WORD2VEC, binary=True, unicode_errors='ignore')

end = time.time()
logging.info('Time to load Word2Vec model: {} seconds'.format(round(end-start, 2)))

# segmenter using the word statistics from Wikipedia
seg = Segmenter(corpus="english")

#verbose=True
source_balanced = 1
balanced = 1

runTransBoostler = True
runRDNB = False
theoryRevision = True

transfer = Transfer(seg)
similarity = Similarity(seg)
revision = TheoryRevision()

def rdnb(background, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, refine=None, transfer=None):
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

def get_confusion_matrix(to_predicate):
    # Get confusion matrix by reading results from db files created by the Java application
    logging.info('Converting results file to txt')

    if not os.path.exists(params.TEST_OUTPUT):
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

def main():
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
          
    for experiment in experiments:

        # Dictionaries to keep all experiments results
        transboostler_experiments = {}
        transboostler_confusion_matrix = {}

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

            
            # Get the list of predicates from source tree
            preds, preds_learned = [], []
            preds = list(set(utils.sweep_tree(structured)))
            preds_learned = [pred.replace('.', '').replace('+', '').replace('-', '') for pred in bk[source] if pred.split('(')[0] != predicate and pred.split('(')[0] in preds and 'recursion_' not in pred]

            # Get all rules learned by RDN-B
            refine_structure = utils.get_all_rules_from_tree(structured)
            utils.write_to_file(refine_structure, params.REFINE_FILENAME)
            
            #preds_learned = ['author(class,author)', 'venue(class,venue)', 'samebib(class,class)', 'sameauthor(author,author)', 'sametitle(title,title)', 'samevenue(venue,venue)', 'title(class,title)', 'haswordauthor(author,word)', 'harswordtitle(title,word)', 'haswordvenue(venue,word)']

            # Create word embeddings and calculate similarities
            targets = [t.replace('.', '').replace('+', '').replace('-', '') for t in set(bk[target]) if t.split('(')[0] != to_predicate and 'recursion_' not in t]


            # Build word vectors for source and target predicates
            sources = utils.build_triples(preds_learned)
            targets = utils.build_triples(targets)

        # Set number of folds
        n_folds = datasets.get_n_folds(target)

        transboostler_target_trees, rdnb_target_trees = [], []
        for i in range(n_folds):
            logging.info('\n Starting fold {} of {} folds \n'.format(i+1, n_folds))

            [tar_train_facts, tar_test_facts] =  datasets.load_pre_saved_folds(i+1, target, 'facts')
            [tar_train_pos, tar_test_pos]     =  datasets.load_pre_saved_folds(i+1, target, 'pos')
            [tar_train_neg, tar_test_neg]     =  datasets.load_pre_saved_folds(i+1, target, 'neg')
            
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

                if(runTransBoostler):

                    # FastText Experiment 

                    # Build word vectors
                    fasttext_sources = transfer.build_fasttext_array(sources, fastTextModel, method=params.METHOD)
                    fasttext_targets = transfer.build_fasttext_array(targets, fastTextModel, method=params.METHOD)
                    
                    similarities = similarity.cosine_similarities(fasttext_sources, fasttext_targets)
                    
                    transboostler_experiments['fasttext'] = {}
                    transboostler_experiments['fasttext']['cosine'] = {key: {'CLL': [], 'AUC ROC': [], 'AUC PR': [], 'Learning Time': [], 'Inference Time': []} for key in params.AMOUNTS} 

                    transboostler_confusion_matrix['fasttext'] = {}
                    transboostler_confusion_matrix['fasttext']['cosine'] = {key: {'TP': [], 'FP': [], 'TN': [], 'FN': []} for key in params.AMOUNTS} 

                    logging.info('Searching for similarities \n')
                    
                    # Map source predicates to targets and creates transfer file
                    mapping = transfer.map_predicates(preds_learned, similarities, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)
                    transfer.write_to_file_closest_distance(predicate, to_predicate, arity, mapping, 'experiments/' + experiment_title, recursion=recursion, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)

                    # Get testing results

                    logging.info('Training using transfer \n')

                    model, t_results, learning_time, inference_time = rdnb(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, params.REFINE_FILENAME, params.TRANSFER_FILENAME)
                    
                    if(theoryRevision):
                        t_results, learning_time, inference_time = revision.apply(model, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, learning_time, inference_time)

                    transboostler_experiments['fasttext']['cosine'][amount]['CLL'].append(t_results['CLL'])
                    transboostler_experiments['fasttext']['cosine'][amount]['AUC ROC'].append(t_results['AUC ROC'])
                    transboostler_experiments['fasttext']['cosine'][amount]['AUC PR'].append(t_results['AUC PR'])
                    transboostler_experiments['fasttext']['cosine'][amount]['Learning Time'].append(learning_time)
                    transboostler_experiments['fasttext']['cosine'][amount]['Inference Time'].append(inference_time)

                    cm = get_confusion_matrix(to_predicate)
                    transboostler_confusion_matrix['fasttext']['cosine'][amount]['TP'].append(cm['TP']) 
                    transboostler_confusion_matrix['fasttext']['cosine'][amount]['FP'].append(cm['FP']) 
                    transboostler_confusion_matrix['fasttext']['cosine'][amount]['TN'].append(cm['TN']) 
                    transboostler_confusion_matrix['fasttext']['cosine'][amount]['FN'].append(cm['FN']) 
                    del model, cm, t_results, learning_time, inference_time

                    # Word2Vec

                    # Build word vectors
                    word2vec_sources = transfer.build_word2vec_array(sources, word2vecModel, method=params.METHOD)
                    word2vec_targets = transfer.build_word2vec_array(targets, word2vecModel, method=params.METHOD)
                    
                    similarities = similarity.cosine_similarities(word2vec_sources, word2vec_targets)
                    
                    transboostler_experiments['word2vec'] = {}
                    transboostler_experiments['word2vec']['cosine'] = {key: {'CLL': [], 'AUC ROC': [], 'AUC PR': [], 'Learning Time': [], 'Inference Time': []} for key in params.AMOUNTS} 

                    transboostler_confusion_matrix['word2vec'] = {}
                    transboostler_confusion_matrix['word2vec']['cosine'] = {key: {'TP': [], 'FP': [], 'TN': [], 'FN': []} for key in params.AMOUNTS}
                    
                    # Map source predicates to targets and creates transfer file
                    mapping = transfer.map_predicates(preds_learned, similarities, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)
                    transfer.write_to_file_closest_distance(predicate, to_predicate, arity, mapping, 'experiments/' + experiment_title, recursion=recursion, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)

                    # Get testing results

                    logging.info('Training using transfer \n')

                    model, t_results, learning_time, inference_time = rdnb(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_train_pos, tar_test_neg, tar_test_facts, params.REFINE_FILENAME, params.TRANSFER_FILENAME)
                    
                    transboostler_experiments['word2vec']['cosine'][amount]['CLL'].append(t_results['CLL'])
                    transboostler_experiments['word2vec']['cosine'][amount]['AUC ROC'].append(t_results['AUC ROC'])
                    transboostler_experiments['word2vec']['cosine'][amount]['AUC PR'].append(t_results['AUC PR'])
                    transboostler_experiments['word2vec']['cosine'][amount]['Learning Time'].append(learning_time)
                    transboostler_experiments['word2vec']['cosine'][amount]['Inference Time'].append(inference_time)

                    cm = get_confusion_matrix(to_predicate)
                    transboostler_confusion_matrix['word2vec']['cosine'][amount]['TP'].append(cm['TP']) 
                    transboostler_confusion_matrix['word2vec']['cosine'][amount]['FP'].append(cm['FP']) 
                    transboostler_confusion_matrix['word2vec']['cosine'][amount]['TN'].append(cm['TN']) 
                    transboostler_confusion_matrix['word2vec']['cosine'][amount]['FN'].append(cm['FN']) 
                    del model, cm, t_results, learning_time, inference_time

                if(runRDNB):

                    # Dictionary to keep amounts values
                    RDNB_results = {key: {'CLL': [], 'AUC ROC': [], 'AUC PR': [], 'Learning Time': [], 'Inference Time': []} for key in params.AMOUNTS}
                    RDNB_confusion_matrix = {key: {'TP': [], 'FP': [], 'TN': [], 'FN': []} for key in params.AMOUNTS}

                    logging.info('Starting to learn from scratch')

                    
                    # Get testing results
                    model, t_results, learning_time, inference_time = rdnb(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_train_pos, tar_test_neg, tar_test_facts)


                    # Get target trees
                    structured = []
                    for i in range(params.TREES):
                        structured.append(model.get_structured_tree(treenumber=i+1).copy())

                    RDNB_results[amount]['CLL'].append(t_results['CLL'])
                    RDNB_results[amount]['AUC ROC'].append(t_results['AUC ROC'])
                    RDNB_results[amount]['AUC PR'].append(t_results['AUC PR'])
                    RDNB_results[amount]['Learning Time'].append(t_results['Learning Time'])
                    RDNB_results[amount]['Learning Time'].append(t_results['Inference Time'])

                    cm = get_confusion_matrix(to_predicate)
                    RDNB_confusion_matrix[amount]['TP'].append(cm['TP'])
                    RDNB_confusion_matrix[amount]['FP'].append(cm['FP'])
                    RDNB_confusion_matrix[amount]['TN'].append(cm['TN'])
                    RDNB_confusion_matrix[amount]['FN'].append(cm['FN'])

                    results = {}
                    results = utils.get_results_dict(t_results, learning_time, inference_time)
                    utils.show_results(results)
                    del model, cm, t_results, learning_time, inference_time, results

                    del model

        if(runTransBoostler):

            # Save all results using transfer
            utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/transboostler_confusion_matrix.json'.format(_id, source, target), transboostler_confusion_matrix)
            utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/transboostler_curves_folds.json'.format(_id, source, target), transboostler_experiments)           

        if(runRDNB):
            
            # Save all CV results
            utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/rdnb_curves_folds.json'.format(_id, source, target), rdnb_folds_results)
            utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/rdnb_confusion_matrix.json'.format(_id, source, target), rdnb_confusion_matrix)

if __name__ == '__main__':
    sys.exit(main())
