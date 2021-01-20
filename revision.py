
from boostsrl import boostsrl
import parameters as params
import utils as utils
import copy
import time

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler("app.log"),logging.StreamHandler()])

class TheoryRevision:

	def __init__(self):
		pass

	def get_refine_file(struct, forceLearning=False, treenumber=1):
		'''Generate the refine file from given tree structure'''
		target = struct[0]
		nodes = struct[1]

		tree = treenumber-1
		refine = []
		for path, value in nodes.items():
			node = target + ' :- ' + value + '.' if not path else value + '.'
			branchTrue = 'true' if revision.get_branch_with(path, 'true') in nodes or forceLearning else 'false'
			branchFalse = 'true' if revision.get_branch_with(path, 'false') in nodes or forceLearning else 'false'
			refine.append(';'.join([str(tree), path, node, branchTrue, branchFalse]))
		return refine

	def get_candidate(structure, variances, treenumber=1, no_pruning=False):
		'''Get candidate refining every revision point in a tree'''
		target = structure[0]
		nodes = structure[1]
		leaves = structure[2]
		if '' not in nodes:
			return []
		tree = self.get_tree(nodes, leaves, variances)
		gen = self.generalize_tree(tree) if not no_pruning else tree
		new_struct = self.get_structured_from_tree(target, gen)
		return self.get_refine_file(new_struct, forceLearning=True, treenumber=treenumber)

	def get_boosted_candidate(self, structure, variances, no_pruning=False):
		refine = []
		for i in range(len(structure)):
			refine += revision.get_candidate(structure[i], variances[i], i+1, no_pruning=no_pruning)
		return refine

	def apply(self, model, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, learning_time, inference_time):

		total_revision_time = 0
		best_cll = - float('inf')
		best_structured = None
		best_model_results = None
		pl_t_results = 0

		# refine candidates
		structured = []
		for i in range(params.TREES):
			structured.append(model.get_structured_tree(treenumber=i+1).copy())

        # Get variances for revision theory 
		variances = [model.get_variances(treenumber=i+1) for i in range(params.TREES)]

        # Test using training set
		start = time.time()
		results = boostsrl.test(model, train_pos, train_neg, train_facts, trees=params.TREES)
		end = time.time()
		t_results = results.summarize_results()
		best_model_cll = t_results['CLL']
		best_model_results = copy.deepcopy(t_results)
		inference_time = end-start
		total_revision_time = learning_time + inference_time
		best_model_structured = copy.deepcopy(structured)

		logging.info('Parameter learned model CLL:{}'.format(t_results['CLL']))
		logging.info('Strucuture after Parameter Learning \n')
        
		for w in structured:
			logging.info(w)
		for v in variances:
			logging.info(v)
		logging.info('\n')

		utils.save_best_model_files()

		logging.info('******************************************')
		logging.info('Performing Theory Revision')
		logging.info('******************************************')

		for i in range(params.MAX_REVISION_ITERATIONS):
			logging.info('Refining iteration {}'.format(str(i+1)))
			logging.info('********************************')
			found_better = False
			candidate = self.get_boosted_candidate(best_model_structured, variances)

			if not len(candidate):
                # Perform revision without pruning
				logging.info('Pruning resulted in null theory\n')
				candidate = self.get_boosted_candidate(best_model_structured, variances, no_pruning=True)

			logging.info('Candidate for revision')
			for item in candidate:
				logging.info(item)
			logging.info('\n')

			logging.info('Refining candidate')
			logging.info('***************************')

			utils.write_to_file(candidate, params.REFINE_FILENAME)
			model, t_results, learning_time, inference_time = self.rdnb(background, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, params.REFINE_FILENAME, params.TRANSFER_FILENAME)

            # Test using training set
			results = boostsrl.test(model, part_tar_train_pos, part_tar_train_neg, tar_train_facts, trees=params.TREES)
			t_results = results.summarize_results()

			structured = []
			for i in range(trees):
				structured.append(model.get_structured_tree(treenumber=i+1).copy())

			total_revision_time = total_revision_time + learning_time + inference_time
            
			if t_results['CLL'] > best_model_cll:
				found_better = True
				best_model_cll = t_results['CLL']
				best_model_structured = copy.deepcopy(structured)
				best_model_results = copy.deepcopy(t_results)
				utils.save_model_files()

			logging.info('Refined model CLL: %s' % t_results['CLL'])
			logging.info('\n')
			if found_better == False:
				break

		# set total revision time to t_results learning time
		best_model_results['Learning time'] = total_revision_time

		logging.info('******************************************')
		logging.info('Best model found')
		logging.info('******************************************')

		utils.show_results(utils.get_results_dict(best_model_results, learning_time, inference_time))

		utils.delete_folder(params.TRAIN_FOLDER_FILES[:-1])
		utils.delete_folder(params.TEST_FOLDER_FILES[:-1])

		utils.delete_file(params.TRAIN_OUTPUT_FILE)
		utils.delete_file(params.TEST_OUTPUT_FILE)

		logging.info('Total revision time: %s' % total_revision_time)
		logging.info('Best scored revision CLL: %s' % best_cll)
		logging.info('\n')

		return best_model_results, total_revision_time, inference_time

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
