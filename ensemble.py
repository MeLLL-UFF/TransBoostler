
from experiments import experiments, bk, setups
from os.path import isfile, join
import parameters as params
from os import listdir
import utils as utils
import pandas as pd
import pickle
import re

class VotingSchemes:

    def __init__(self, experiment_type):
    	self.experiment_type = experiment_type

    def __get_weight(self,maximum,minimum,current):
		#return (current-minimum)/(maximum-minimum)
        return 1-(current)/(maximum)

    def __get_weighted_vote(self,maximun,minimum,euclidean=0,wmd=0,relax_wmd=0):
        denominador = sum([euclidean,wmd,relax_wmd])
        numerador = get_weight(denominador,minimum,euclidean)*euclidean + get_weight(denominador,minimum,wmd)*wmd + get_weight(denominador,minimum,relax_wmd)*relax_wmd  # + get_weight(maximun,minimum,softcosine)*softcosine
        return numerador/denominador

    def create_ballot(self,ranked_mappings):
        votes = {}
        for line in ranked_mappings:
            if 'setParam' in line or 'setMap' in line or not line:
                continue

            source,targets = line.split(':')[0].strip(), line.split(':')[1]
            targets = re.split(r',\s*(?![^()]*\))', targets)
            n = len(targets)-1
            for target in targets:
                votes[source + ',' + target.strip()] = n
                n -= 1
        return votes

    def borda_count(self,ballots):
        count = {}
        
        for key in ballots[0].keys():
            count[key] = 0

        # Ballots são os votos de todas as ditâncias
        for ballot in ballots:
            for key in ballot.keys():
                count[key] += ballot[key]
        return count

    def borda_count_voting(self,sources,experiment_title,embeddingModel):
        ballots = []

        for sim in ['euclidean', 'softcosine', 'wmd']:
            file = open(params.ROOT_PATH + '{}-experiments/{}/transfer_{}_{}.txt'.format(self.experiment_type,experiment_title,embeddingModel,sim),'r').read().split('\n')
            ballots.append(self.create_ballot(file))

        bordaCount = pd.DataFrame.from_dict(self.borda_count(ballots), orient='index', columns=['votes']).rename_axis('candidates').sort_values(by=['votes', 'candidates'], ascending=[False, True])

        mappings = {}
        targets_taken = []
        indexes = bordaCount.index.tolist()

        for source in sources:
        	mappings[source] = []

        for index in indexes:
            index = re.split(r',\s*(?![^()]*\))', index)
            source, target = index[0].rstrip().replace('`', ''), index[1].rstrip().replace('`', '')

            if 'recursion_' in source or mappings[source]:
            	continue

            if(target.split('(')[0] not in targets_taken):
            	mappings[source].append(target)
            	targets_taken.append(target.split('(')[0])

        return mappings

    def majority_vote(self,sources,experiment_title,embeddingModel):

        choices = {}
        for sim in ['euclidean', 'softcosine', 'wmd']:
            file = open(params.ROOT_PATH + '{}-experiments/{}/transfer_{}_{}.txt'.format(self.experiment_type,experiment_title,embeddingModel,sim),'r').read().split('\n')
    		
            for line in file:
                if 'setParam' in line or 'setMap' in line or not line:
                    continue

                source,targets = line.split(':')[0].strip(), line.split(':')[1]

                if source not in choices:
                    choices[source] = []

                target = re.split(r',\s*(?![^()]*\))', targets)[0]
                #for target in targets:
                choices[source].append(target.strip())
        
        mappings = {}
        targets_taken = []
        chosen = False

        for source in sources:

            #if 'setParam' in line or 'setMap' in line or not line:
            #    continue

            if source not in mappings:
                mappings[source] = []
            else:
            	continue

            #source = line.split(':')[0].strip()
            if source not in choices:
            	continue

            unique_targets = list(set(choices[source]))[:]
            targets = choices[source][:]

            max_count = 0
            current = ''

            vots = []
            for target in unique_targets:
                
                if target not in targets_taken:
                    
                    current_count = targets.count(target)
                    vots.append(current_count)
                    
                    if max_count < current_count:
                        max_count = current_count
                        current = target
                        chosen = True
            
            if chosen:
                if vots.count(vots[0]) == len(vots):
                     
                    # Se todos tem o mesmo número de votos, escolhe usando ordem alfabética
                    targets.sort()

                    for target in targets:
                        if target not in targets_taken:
                            current = target
                            break

                mappings[source] = [current]
                targets_taken.append(current)
                chosen = False
        return mappings



def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def weight_voting():
	for experiment in experiments:

		_id = experiment['id']
		source = experiment['source']
		target = experiment['target']
		predicate = experiment['predicate']
		to_predicate = experiment['to_predicate']
		arity = experiment['arity']

		experiment_title = _id + '_' + source + '_' + target

		print(experiment_title + '\n')

		# Get sources and targets
		sources = [s.replace('.', '').replace('+', '').replace('-', '').split('(')[0] for s in set(bk[source]) if s.replace('`','').split('(')[0] != predicate and 'recursion_' not in s]
		targets = [t.replace('.', '').replace('+', '').replace('-', '').split('(')[0] for t in set(bk[target]) if t.replace('`','').split('(')[0] != to_predicate and 'recursion_' not in t]

		sources,targets = list(set(sources)),list(set(targets))
		ja_foi = []

		nodes = load_pickle_file(params.ROOT_PATH +  'resources/{}_{}_{}/{}'.format(_id, source, target, params.SOURCE_TREE_NODES_FILES))
		sources_dict =  utils.match_bk_source(set(bk[source]))
		sources = [sources_dict[node].split('(')[0] for node in utils.sweep_tree(nodes, preds=[]) if node != predicate and 'recursion_' not in node]

		mappings = {}
		targets_taken = []

		for source in sources:

			if source in ja_foi:
				continue
			ja_foi.append(source)

			similarities = pd.DataFrame()
			
			for similarityMetric in ['euclidean', 'wmd', 'relax-wmd']:

				path = params.ROOT_PATH + '{}-experiments/{}/similarities/fasttext/{}/{}_similarities.csv'.format(self.experiment_type,experiment_title,similarityMetric,source)
				try:
					current = pd.read_csv(path).set_index('candidates').rename(columns={'similarity': similarityMetric})
				except FileNotFoundError:
					#print(f"Could not find {path}")
					continue
				sim = current[similarityMetric]
				similarities.insert(0, similarityMetric, sim)
			
			ensemble = {}
			for index,row in similarities.iterrows():
				try:
					_all = [row[similarityMetric] for similarityMetric in ['euclidean', 'wmd', 'relax-wmd']]
					maximum = max(_all)
					minimum = min(_all)

					ensemble[index] = [get_weighted_vote(maximum,minimum,row['euclidean'],row['wmd'],row['relax-wmd'])]
				except KeyError:
					#print(f"Could not process {index}")
					continue
			if not ensemble:
				continue
			ensemble_dt = pd.DataFrame.from_dict(ensemble, orient='index', columns=['similarity']).rename_axis('candidates').sort_values(by=['similarity', 'candidates'], ascending=[False, True])
			
			indexes = ensemble_dt.index.tolist()

			for index in indexes:
				index = re.split(r',\s*(?![^()]*\))', index)
				source, target = index[0].split('(')[0].rstrip(), index[1].split('(')[0].rstrip()

				if source in mappings:
					continue

				if(target not in targets_taken):
					targets_taken.append(target)
					mappings[source] = target

		print(mappings)

		print('\n\n\n')
		
# v = VotingSchemes()
# print(v.borda_count_voting('1_imdb_uwcse'))
# # print(v.majority_vote())
		
		
