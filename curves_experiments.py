
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
#logging.getLogger().addHandler(logging.StreamHandler())
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, format='%(message)s')

if not os.path.exists(params.WIKIPEDIA_FASTTEXT):
    raise ValueError("SKIP: You need to download the fasttext wikipedia model")

logging.info('Loading fasttext model')
#fastTextModel = FastText.load_fasttext_format(params.WIKIPEDIA_FASTTEXT)

#if not os.path.exists(params.GOOGLE_WORD2VEC):
#    raise ValueError("SKIP: You need to download the google news model")
#logging.info('Loading word2vec model')
#word2vecModel = api.load(params.GOOGLE_WORD2VEC)

# segmenter using the word statistics from Wikipedia
seg = Segmenter(corpus="english")

#verbose=True
source_balanced = 1
balanced = 1

runTransBoostler = True
runRDNB = False
theoryRevision = True

transfer = Transfer()
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
        transboostler_experiments_curves = {}
        transboostler_confusion_matrix = {}

        # Dataframes to keep all RDN-B experiments results
        rdnb_folds_results = pd.DataFrame([], columns=['CLL', 'AUC ROC', 'AUC PR', 'Precision', 'Recall', 'F1', 'Total Learning Time', 'Total Inference Time'])
        rdnb_confusion_matrix  = pd.DataFrame([], columns=['TP', 'FP', 'TN', 'FN'])

        # Dictionary to keep amounts values
        RDNB_results = {key: {'AUC ROC': 0, 'AUC PR': 0} for key in params.AMOUNTS}

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

            saolspalsoaksoaksop

            for amount in params.AMOUNTS:
                logging.info('Amount of data: ' + str(amount))
                part_tar_train_pos = tar_train_pos[:int(amount * len(tar_train_pos))]
                part_tar_train_neg = tar_train_neg[:int(amount * len(tar_train_neg))]

                # Creating background
                background = boostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=params.MAXTREEDEPTH, nodeSize=params.NODESIZE, numOfClauses=params.NUMOFCLAUSES)

                if(runTransBoostler):

                    # Build word vectors for source and target predicates
                    sources = utils.build_triples(preds_learned)
                    targets = utils.build_triples(targets)

                    # FastText Experiment 

                    # Build word vectors
                    fasttext_sources = transfer.build_fasttext_array(sources, fastTextModel, method=params.METHOD)
                    fasttext_targets = transfer.build_fasttext_array(targets, fastTextModel, method=params.METHOD)
                    
                    similarities = similarity.cosine_similarities(fasttext_sources, fasttext_targets)
                    
                    transboostler_experiments['fasttext'] = {}
                    transboostler_experiments['fasttext']['cosine'] = {}
                    transboostler_experiments_curves['fasttext'] = {key: {'AUC ROC': [], 'AUC PR': []} for key in params.AMOUNTS} 

                    transboostler_confusion_matrix['fasttext'] = {}
                    transboostler_confusion_matrix['fasttext']['cosine'] = {}

                    logging.info('Searching for similarities \n')
                    
                    # Map source predicates to targets and creates transfer file
                    mapping = transfer.map_predicates(preds_learned, similarities, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)
                    transfer.write_to_file_closest_distance(predicate, to_predicate, arity, mapping, 'experiments/' + experiment_title, recursion=recursion, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)

                    # Get testing results

                    logging.info('Training using transfer \n')

                    model, t_results, learning_time, inference_time = rdnb(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, params.REFINE_FILENAME, params.TRANSFER_FILENAME)
                    
                    if(theoryRevision):
                        t_results, learning_time, inference_time = revision.apply(model, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts)

                    transboostler_experiments_curves['fasttext'][amount]['AUC ROC'].append(t_results['AUC ROC'])
                    transboostler_experiments_curves['fasttext'][amount]['AUC PR'].append(t_results['AUC PR'])

                    if(amount == 1.0):

                        results = {}
                        results = utils.get_results_dict(t_results, learning_time, inference_time)
                        transboostler_experiments['fasttext']['cosine'] = results

                        utils.show_results(results)

                        transboostler_confusion_matrix['fasttext']['cosine'] = get_confusion_matrix(to_predicate)

                        del results
                    del model

                    # Word2Vec

                    # Build word vectors
                    word2vec_sources = transfer.build_word2vec_array(sources, word2vecModel, method=params.METHOD)
                    word2vec_targets = transfer.build_word2vec_array(targets, word2vecModel, method=params.METHOD)
                    
                    similarities = similarity.cosine_similarities(word2vec_sources, word2vec_targets)
                    
                    transboostler_experiments['word2vec'] = {}
                    transboostler_experiments['word2vec']['cosine'] = {}
                    transboostler_experiments_curves['word2vec'] = {key: {'AUC ROC': [], 'AUC PR': []} for key in params.AMOUNTS} 

                    transboostler_confusion_matrix['word2vec'] = {}
                    transboostler_confusion_matrix['word2vec']['cosine'] = {}
                    
                    # Map source predicates to targets and creates transfer file
                    mapping = transfer.map_predicates(preds_learned, similarities, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)
                    transfer.write_to_file_closest_distance(predicate, to_predicate, arity, mapping, 'experiments/' + experiment_title, recursion=recursion, allowSameTargetMap=params.ALLOW_SAME_TARGET_MAP)

                    # Get testing results

                    logging.info('Training using transfer \n')

                    t_results, learning_time, inference_time = rdnb(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_train_pos, tar_test_neg, tar_test_facts, params.REFINE_FILENAME, params.TRANSFER_FILENAME)
                    
                    transboostler_experiments_curves['word2vec'][amount]['AUC ROC'].append(t_results['AUC ROC'])
                    transboostler_experiments_curves['word2vec'][amount]['AUC PR'].append(t_results['AUC PR'])

                    if(amount == 1.0):

                        results = {}
                        results = utils.get_results_dict(t_results, learning_time, inference_time)
                        transboostler_experiments['word2vec']['cosine'] = results

                        utils.show_results(results)

                        transboostler_confusion_matrix['word2vec']['cosine'] = get_confusion_matrix(to_predicate)

                        del results
                    del model

                if(runRDNB):

                    logging.info('Starting to learn from scratch')

                    
                    # Get testing results
                    model, t_results, learning_time, inference_time = rdnb(background, part_tar_train_pos, part_tar_train_neg, tar_train_facts, tar_train_pos, tar_test_neg, tar_test_facts)


                    RDNB_results[amount]['AUC ROC'] += t_results['AUC ROC']
                    RDNB_results[amount]['AUC PR']  += t_results['AUC PR']

                    if(amount == 1.0):

                        results = {}
                        results = utils.get_results_dict(t_results, learning_time, inference_time)

                        utils.show_results(results)
                        
                        rdnb_folds_results = rdnb_folds_results.append(results, ignore_index=True)

                        # Get target trees
                        structured = []
                        for i in range(params.TREES):
                          structured.append(model.get_structured_tree(treenumber=i+1).copy())

                        refine_structure = utils.get_all_rules_from_tree(structured)
                        rdnb_target_trees += refine_structure

                        rdnb_confusion_matrix = rdnb_confusion_matrix.append(get_confusion_matrix(to_predicate), ignore_index=True)

                        del results

                    del model

        if(runTransBoostler):

            # Save all results using transfer
            utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/transboostler_folds.txt'.format(_id, source, target), transboostler_experiments_curves)
            utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/transboostler_confusion_matrix.txt'.format(_id, source, target), transboostler_confusion_matrix)
            utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/transboostler_curves.txt'.format(_id, source, target), transboostler_experiments_curves)           

        if(runRDNB):
            
            # Save all CV results
            rdnb_folds_results.to_csv(os.getcwd() + '/experiments/{}_{}_{}/rdnb_folds.txt'.format(_id, source, target))
            rdnb_confusion_matrix.to_csv(os.getcwd() + '/experiments/{}_{}_{}/rdnb_confusion_matrix.txt'.format(_id, source, target), index=False)

            all_RDNB_results = pd.DataFrame.from_dict(RDNB_results)
            all_RDNB_results = all_RDNB_results/n_folds
            
            all_RDNB_results.to_csv(os.getcwd() + '/experiments/{}_{}_{}/RDNB_curves.csv'.format(_id, source, target))

if __name__ == '__main__':
    sys.exit(main())
