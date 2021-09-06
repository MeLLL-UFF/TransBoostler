
import re
import pandas as pd
from experiments import experiments, bk, setups
from os import listdir, getcwd, mkdir
from os.path import isfile, join, exists


for experiment in experiments:

    experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
    target = experiment['target']

    print('Starting experiment {} \n'.format(experiment_title))

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

    for setup in setups: 
        mappings, targets_taken, sources = {}, {}, []
        embeddingModel = setup['model'].lower()
        similarityMetric = setup['similarity_metric'].lower()

        path = getcwd() + '/experiments/{}/similarities/fasttext/{}'.format(experiment_title,similarityMetric)
        files = [path + '/' + f for f in listdir(path) if isfile(join(path, f))]

        dt = pd.read_csv(files[0])
        for file in files[1:]:
        	temp = pd.read_csv(file)
        	dt = pd.concat([dt, temp], ignore_index=True)

        dt = dt.sort_values(by=['similarity'])
        combos = dt['candidates'].tolist()

        for combo in combos:
        	combo = re.split(r',\s*(?![^()]*\))', combo)
        	source, target = combo[0].rstrip(), combo[1].rstrip()
        	sources.append(source)

        	if(source in mappings or target in targets_taken):
        		continue

        	mappings[source] = target
        	targets_taken[target] = 1

        for source in sources:
        	if source not in mappings:
        		mappings[source] = 'empty'
       	resp = pd.DataFrame.from_dict(mappings, orient='index')

       	path = getcwd() + '/resources/{}/'.format(experiment_title)
       	if not exists(path):
       		mkdir(path)

        dt.to_csv(getcwd() + '/resources/{}/most_similar_{}'.format(experiment_title,similarityMetric))
       	resp.to_csv(getcwd() + '/resources/{}/most_similar_{}'.format(experiment_title,similarityMetric))

    del dt, combos



        
