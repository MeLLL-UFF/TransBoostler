
from boostsrl import boostsrl
import parameters as params
import utils as utils
import copy
import math
import time

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler("app.log"),logging.StreamHandler()])

class TheoryRevision:

	def __init__(self):
		pass

	def get_branch_with(self, branch, next_branch):
		'''Append next_branch at branch'''
		if not branch:
			return next_branch
		b = branch.split(',')
		b.append(next_branch)
		return ','.join(b)

	def get_structured_from_tree_helper(self, path, root, nodes, leaves):
		if isinstance(root, list):
			leaves[path] = root
		elif isinstance(root, dict):
			i = list(root.keys())[0]
			value = root[i]
			children= value[1]
			split = [] if path == '' else path.split(',')
			left, right = ','.join(split+['true']), ','.join(split+['false'])
			nodes[path] = i
			self.get_structured_from_tree_helper(left, children[0], nodes, leaves)
			self.get_structured_from_tree_helper(right, children[1], nodes, leaves)

	def get_structured_from_tree(self, target, tree):
		nodes, leaves = {}, {}
		self.get_structured_from_tree_helper('', tree, nodes, leaves)
		return [target, nodes, leaves]

	def get_tree_helper(self, path, nodes, leaves, variances, no_variances=False):
		children = [None, None]
		split = [] if path == '' else path.split(',')
		left, right = ','.join(split+['true']), ','.join(split+['false'])
		varc = variances[path] if not no_variances else []
		if left in nodes:
			children[0] = self.get_tree_helper(left, nodes, leaves, variances, no_variances=no_variances)
		if right in nodes:
			children[1] = self.get_tree_helper(right, nodes, leaves, variances, no_variances=no_variances)
		if left in leaves:
			children[0] = leaves[left] # { 'type': 'leaf', 'std_dev': leaves[left][0], 'neg': leaves[left][1], 'pos': leaves[left][2] }
		if right in leaves:
			children[1] = leaves[right]
		return { nodes[path]: [varc, children] }

	def get_tree(self, nodes, leaves, variances, no_variances=False):
		return self.get_tree_helper('', nodes, leaves, variances, no_variances=no_variances)

	def generalize_tree_helper(self, root):
		if isinstance(root, list):
			return root
		elif isinstance(root, dict):
			i = list(root.keys())[0]
			value = root[i]
			children= value[1]
			variances = value[0]
			true_child, false_child = self.generalize_tree_helper(children[0]), self.generalize_tree_helper(children[1])

			# if TRUE child has 0 examples reached
			if math.isnan(variances[0]):
				return false_child
			# if FALSE child has 0 examples reached
			if math.isnan(variances[1]):
				return true_child
			# if node has only leaves
			if isinstance(true_child, list) and isinstance(false_child, list):
				if variances[0] >= 0.0025 and variances[1] >= 0.0025:
					return [0, true_child[1] + false_child[1], true_child[2] + false_child[2]] # return a leaf
			# otherwise
			return { i: [variances, [true_child, false_child]] }

	def generalize_tree(self, tree):
		ntree = copy.deepcopy(tree)
		return self.generalize_tree_helper(ntree)

	def get_refine_file(self, struct, forceLearning=False, treenumber=1):
		'''Generate the refine file from given tree structure'''
		target = struct[0]
		nodes = struct[1]

		tree = treenumber-1
		refine = []
		for path, value in nodes.items():
			node = target + ' :- ' + value + '.' if not path else value + '.'
			branchTrue = 'true' if self.get_branch_with(path, 'true') in nodes or forceLearning else 'false'
			branchFalse = 'true' if self.get_branch_with(path, 'false') in nodes or forceLearning else 'false'
			refine.append(';'.join([str(tree), path, node, branchTrue, branchFalse]))
		return refine

	def get_candidate(self, structure, variances, treenumber=1, no_pruning=False):
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
			refine += self.get_candidate(structure[i], variances[i], i+1, no_pruning=no_pruning)
		return refine


	def get_boosted_refine_file(self, structs, forceLearning=False):
		refine = []
		for i in range(len(structs)):
			refine += self.get_refine_file(structs[i], treenumber=i+1, forceLearning=forceLearning)
		return refine

	def apply(self, background, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, source_structure):
		'''Function responsible for starting the theory revision process'''

		total_revision_time = 0
		best_cll = - float('inf')
		best_structured = None
		best_model_results = None
		pl_t_results = 0

		# Parameter learning
		logging.info('******************************************')
		logging.info('Performing Parameter Learning')
		logging.info('******************************************')
		logging.info('Refine')
		for item in self.get_boosted_refine_file(source_structure):
			logging.info(item)
		logging.info('\n')

		model, t_results, learning_time, inference_time = self.train_and_test(background, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, refine=params.REFINE_FILENAME, transfer=params.TRANSFER_FILENAME)
		pl_t_results = copy.deepcopy(t_results)

		structured = []
		for i in range(params.TREES):
			structured.append(model.get_structured_tree(treenumber=i+1).copy())

		variances = [model.get_variances(treenumber=i+1) for i in range(params.TREES)]

        # Test using training set - Score model
		start = time.time()
		results = boostsrl.test(model, train_pos, train_neg, train_facts, trees=params.TREES)
		scored_results = results.summarize_results()
		end = time.time()
		inference_time = end-start

		best_model_cll = scored_results['CLL']
		best_model_results = copy.deepcopy(t_results)

		total_revision_time = learning_time + inference_time

		logging.info('Parameter learned model CLL:{} \n'.format(scored_results['CLL']))
		logging.info('Strucuture after Parameter Learning \n')

		best_model_structured = copy.deepcopy(structured)
		logging.info('Structure after Parameter Learning')

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

			utils.write_to_file(candidate, params.REFINE_REVISION_FILENAME)
			model, t_results, learning_time, inference_time = self.train_and_test(background, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, refine=params.REFINE_REVISION_FILENAME)

			structured = []
			for i in range(params.TREES):
				structured.append(model.get_structured_tree(treenumber=i+1).copy())

			variances = [model.get_variances(treenumber=i+1) for i in range(params.TREES)]

			# Inference on the training set to catch where it can be improved
			start = time.time()
			results = boostsrl.test(model, train_pos, train_neg, train_facts, trees=params.TREES)
			scored_results = results.summarize_results()
			end = time.time()
			inference_time = end-start

			total_revision_time = total_revision_time + learning_time + inference_time

			if scored_results['CLL'] > best_model_cll:
				found_better = True
				best_model_cll = scored_results['CLL']
				best_model_structured = copy.deepcopy(structured)
				best_model_results = copy.deepcopy(t_results)
				utils.save_best_model_files()

			logging.info('Refined model CLL: %s' % scored_results['CLL'])
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
		logging.info('Best scored revision CLL: %s' % best_model_cll)
		logging.info('\n')

		return best_model_results, total_revision_time, inference_time

	def train_and_test(self, background, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, refine=None, transfer=None):
		'''
	        Train RDN-B using transfer learning
	    '''

		start = time.time()
		model = boostsrl.train(background, train_pos, train_neg, train_facts, refine=refine, transfer=transfer, trees=params.TREES)
	    
		end = time.time()
		learning_time = end-start

		logging.info('Model training time {}'.format(learning_time))

		will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(params.TREES)]
		for w in will:
			logging.info(w)

		start = time.time()

		# Test transfered model
		results = boostsrl.test(model, test_pos, test_neg, test_facts, trees=params.TREES)

		end = time.time()
		inference_time = end-start

		logging.info('Inference time using transfer learning {}'.format(inference_time))

		return model, results.summarize_results(), learning_time, inference_time
