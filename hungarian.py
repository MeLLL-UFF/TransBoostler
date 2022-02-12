
from scipy.optimize import linear_sum_assignment
import parameters as params
import utils as utils
import pandas as pd
import re

class Hungarian:

	def __init__(self, similarities):
		self.similarities = similarities

	def __get_target_indexes(self,arity=2):
		"""
			Maps the index of targets predicates accordingly to the matrix columns

        	Returns:
            	a dictionary containing the index of a given key
            	a dicitonary containing the target of a given index
		"""

		columns = []
		column_indexes = {}
		index_to_target = {}
		
		indexes = self.similarities.index.tolist()
		for index in indexes:
			index = re.split(r',\s*(?![^()]*\))', index)
			source, target = index[0].rstrip(), index[1].rstrip()

			# Check if predicates are of the same arity
			if(not len(utils.get_all_literals([source])) == arity):
				continue

			if target not in columns:
				columns.append(target)

		for column in columns:
			column_indexes[column] = len(column_indexes)

		for key in column_indexes:
			index_to_target[column_indexes[key]] = key

		return column_indexes, index_to_target

	def __build_cost_matrix(self, arity=2):
		"""
			Returns cost matrix of the bipartite graph given the dataframe of similarities.
			Vertex i of the first partite set (a source predicate) and vertex j of the second set (a target predicate).

			Args:
	            similarities(DataFrame): similarities between (source, target) pairs of predicates
        	Returns:
            	cost matrix of the bipartite graph
		"""
		
		cost_matrix = []

		indexes = self.similarities.index.tolist()
		for index in indexes:
			index = re.split(r',\s*(?![^()]*\))', index)
			source, target = index[0].rstrip(), index[1].rstrip()

			# Check if predicates are of the same arity
			if(not len(utils.get_all_literals([source])) == arity):
				continue

			if source not in self.candidates:
				self.candidates[source] = [target]
			else:
				self.candidates[source].append(target)

		similarities_dictionary = self.similarities.to_dict()['similarity']

		for i in self.candidates:
			row = []
			for j in self.candidates[i]:
				
				# Check if predicates are of the same arity
				if(not (len(utils.get_all_literals([i])) == arity)):
					continue

				row = [0]*len(self.target_ind)
				row[self.target_ind[j]] = similarities_dictionary[i + ',' + j]
				self.source_ind[len(cost_matrix)] = i

			cost_matrix.append(row)

		return cost_matrix

	def __get_targets(self, col_ind):
		"""
			Returns the corresponding mapping to a source predicate given the index assigment

			Args:
	            col_ind(list): list of columns
        	Returns:
            	dictionary of mappings provided by the Hungarian algorithm
		"""
		current_mappings = {}
		for i in range(len(col_ind)):
			source_predicate = self.source_ind[i]
			current_mappings[source_predicate] = [self.ind_to_target[col_ind[i]]]
		return current_mappings

	def assigment(self):
		"""
			Returns columns indexes after solving the linear sum assignment problem using the Hungarian algorithm. 
			Each row is assignment to at most one column, and each column to at most one row.

        	Returns:
            	An array of row indices and one of corresponding column indices giving the optimal assignment. 
		"""
		mappings = {}

		# Assuming we only have predicates of arity one and two
		for i in range(1,3):
			self.candidates = {}
			self.source_ind = {}

			self.target_ind, self.ind_to_target = self.__get_target_indexes(arity=i)
			cost_matrix = self.__build_cost_matrix(arity=i)
			
			if(not cost_matrix):
				continue

			row_ind, col_ind = linear_sum_assignment(cost_matrix)
			mappings.update(self.__get_targets(col_ind))
		return mappings

# similarities = pd.DataFrame()
# for predicate in ['athleteledsportsteam', 'athleteplaysforteam','athleteplaysinleague', 'athleteplayssport', 'teamalsoknownas', 'teamplaysagainstteam','teamplaysinleague']:
#  	current = pd.read_csv(params.ROOT_PATH + 'resources/9_nell_sports_nell_finances/rwmd-similarities/{}_similarities.csv'.format(predicate)).set_index('candidates')
#  	similarities = pd.concat([similarities, current])

# import time
# start = time.time()
# hug = Hungarian(similarities)
# #print(similarities)
# print(hug.assigment())
# print(time.time()-start)

# cost = [[4.9586896896362305, 5.472095489501953],[4.785916328430176,5.348875999450684],[4.654685974121094,5.441676139831543],[4.956645488739014,5.603146553039551],[4.749782085418701,5.210647106170654],[4.361474990844727,5.199010848999023],[4.411990165710449,5.244400978088379],[4.829348564147949,5.565415382385254],[4.5297322273254395,5.28743839263916]]
# row_ind, col_ind = linear_sum_assignment(cost_matrix)
# print(col_ind)