
from experiments import experiments, bk, setups
from os.path import isfile, join
import parameters as params
from os import listdir
import utils as utils
import pandas as pd
import collections
import itertools
import pickle
import re,os

def borda(ballot):
    try:
        n = len([c for c in ballot if c.isalpha()])
        score = itertools.count(n, step = -1)
        result = {}

        for group in [[re.split(r',\s*(?![^()]*\))', item)[1]] if item.count(',') > 1 else [item.split(',')[1]] for item in ballot.split('>')]:
            print(group)
            s = sum(next(score) for item in group)/float(len(group))
            print(s)
            for pref in group:
                result[pref] = s
    except:
        print('Empty predicate')
    return result

def tally(ballots):
    result = collections.defaultdict(int)
    for ballot in ballots:
        for pref,score in borda(ballot).items():
            result[pref]+=score
    result = dict(result)
    return result

for experiment in experiments:
    experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
    print(experiment_title)
    
    path = os.getcwd() + '/experiments/' + experiment_title + '/similarities/fasttext'
    files = [f for f in listdir(path + '/euclidean') if isfile(join(path + '/euclidean', f))]

    for file in files:

        print(file.split('_')[0])

        euclidean = '>'.join(pd.read_csv(path + '/euclidean/' + file)['Unnamed: 0'].values.tolist())
        #softcosine = '>'.join(pd.read_csv(path + '/softcosine/' + file)['Unnamed: 0'].values.tolist())
        wmd = '>'.join(pd.read_csv(path + '/wmd/' + file)['Unnamed: 0'].values.tolist())
        relax_wmd = '>'.join(pd.read_csv(path + '/relax-wmd/' + file)['Unnamed: 0'].values.tolist())

        print(pd.DataFrame.from_dict(tally([euclidean, wmd, relax_wmd]), orient="index", columns=['votes']).sort_values(by='votes', ascending=False))
        print('\n')

#Exemplo

#ballots = [['A','B','C','D','E'],
#           ['B','A','C','D','E'],
#           ['C','B','A','E','D']]


#A: 5+4+3 = 12
#B: 2*4+5 = 13
#C: 2*3+5 = 11
#D: 2*2+1 = 5
#E: 2*1+2 = 4


# bd = BordaCount()

# file = open('/home/thais/Documentos/TransBoostler/transfer-experiments/1_imdb_uwcse/transfer_fasttext_euclidean.txt','r').read().split('\n')
# ballot_1 = bd.create_ballot(file)

# file = open('/home/thais/Documentos/TransBoostler/transfer-experiments/1_imdb_uwcse/transfer_fasttext_wmd.txt','r').read().split('\n')
# ballot_2 = bd.create_ballot(file)

# v = bd.borda_count([ballot_1,ballot_2])

# df = pd.DataFrame.from_dict(v, orient="index", columns=['votes']).rename_axis('candidates').sort_values(by=['votes', 'candidates'], ascending=[False, True])

# print(df)