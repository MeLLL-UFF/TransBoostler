
import re,os
import itertools
import collections
import pandas as pd
from os import listdir
from os.path import isfile, join
from experiments import experiments

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
