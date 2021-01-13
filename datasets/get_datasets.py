'''
   Functions to return datasets in file folder
   Name:         get_datasets.py
   Author:       Rodrigo Azevedo
   Updated:      July 22, 2018
   License:      GPLv3
'''

import re
import os
import unidecode
import csv
import math
import random
import pandas as pd
import json
import copy

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class datasets:
    def get_kfold(test_number, folds):
        '''Separate examples into train and test set.
        It uses k-1 folds for training and 1 single fold for testing'''
        train = []
        test = []
        for i in range(len(folds)):
            if i == test_number:
                test += folds[i]
            else:
                train += folds[i]
        return (train, test)

    def get_kfold_separated(test_number, folds):
        train = []
        test = []
        for i in range(len(folds)):
            if i == test_number:
                test = folds[i]
            else:
                train.append(folds[i])
        return (train, test)

    def get_kfold_small(train_number, folds):
        '''Separate examples into train and test set.
        It uses 1 single fold for training and k-1 folds for testing'''
        train = []
        test = []
        for i in range(len(folds)):
            if i == train_number:
                train += folds[i]
            else:
                test += folds[i]
        return (train, test)

    def group_folds(folds):
        '''Group folds in a single one'''
        train = []
        for i in range(len(folds)):
            train += folds[i]
        return train

    def split_into_folds(examples, n_folds=5, seed=None):
        '''For datasets as nell and yago that have only 1 mega-example'''
        temp = list(examples)
        random.seed(seed)
        random.shuffle(temp)
        s = math.ceil(len(examples)/n_folds)
        ret = []
        for i in range(n_folds-1):
            ret.append(temp[:s])
            temp = temp[s:]
        ret.append(temp)
        random.seed(None)
        return ret

    def balance_neg(target, data, size, seed=None):
        '''Receives [facts, pos, neg] and balance neg according to pos'''
        ret = copy.deepcopy(data)
        neg = []
        random.seed(seed)
        random.shuffle(ret)
        ret = ret[:size]
        random.seed(None)
        for entities in ret:
            neg.append(target + '(' + ','.join(entities) + ').')
        return neg

    def get_neg(target, data):
        '''Receives [facts, pos, neg] and return neg'''
        ret = copy.deepcopy(data)
        neg = []
        for entities in ret:
            neg.append(target + '(' + ','.join(entities) + ').')
        return neg

    def generate_neg(target, data, amount=1, seed=None):
        '''Receives [facts, pos, neg] and generates balanced neg examples in neg according to pos'''
        pos = copy.deepcopy(data)
        neg = []
        objects = set()
        subjects = {}
        for entities in pos:
            if entities[0] not in subjects:
                subjects[entities[0]] = set()
            subjects[entities[0]].add(entities[1])
            objects.add(entities[1])
        random.seed(seed)
        target_objects = list(objects)
        for entities in pos:
            key = entities[0]
            for j in range(amount):
                for tr in range(10):
                    r = random.randint(0, len(target_objects)-1)
                    if target_objects[r] not in subjects[key]:
                        neg.append(target + '(' + ','.join([key, target_objects[r]]) + ').')
                        subjects[key].add(target_objects[r])
                        break
        random.seed(None)
        return neg

    def generate_all_neg(target, data):
        '''Receives [facts, pos, neg] and generates neg examples in neg according to pos'''
        pos = copy.deepcopy(data)
        neg = []
        objects = set()
        subjects = {}
        for entities in pos:
            if entities[0] not in subjects:
                subjects[entities[0]] = set()
            subjects[entities[0]].add(entities[1])
            objects.add(entities[1])
            # for same type
            objects.add(entities[0])
        for entities in pos:
            key = entities[0]
            target_objects = objects.difference(subjects[key])
            for objc in target_objects:
                neg.append(target + '(' + ','.join([key, objc]) + ').')
        return neg

    def get_json_dataset(dataset):
        '''Load dataset from json'''
        with open(os.path.join(__location__, 'files/json/' + dataset + '.json')) as data_file:
            data_loaded = json.load(data_file)
        return data_loaded

    def load(dataset, bk, target=None, seed=None, balanced=1):
        '''Load dataset from json and accept only predicates presented in bk'''
        pattern = '^(\w+)\(.*\).$'
        accepted = set()
        for line in bk:
            m = re.search(pattern, line)
            if m:
                relation = re.sub('[ _]', '', m.group(1))
                accepted.add(relation)
        data = datasets.get_json_dataset(dataset)
        facts = []
        pos = []
        neg = []
        for i in range(len(data[0])): #positives
            facts.append([])
            pos.append([])
            neg.append([])
            for relation, value in data[0][i].items():
                if relation in accepted:
                    if relation == target:
                        for example in value:
                            pos[i].append(relation + '(' + ','.join(example)+ ').')
                            facts[i].append('recursion_' + relation + '(' + ','.join(example)+ ').')
                    else:
                        for example in value:
                            facts[i].append(relation + '(' + ','.join(example)+ ').')
        if target:
            for i in range(len(data[1])): #negatives
                if target in data[1][i]:
                    value = data[1][i][target]
                    if balanced:
                        neg[i] = datasets.balance_neg(target, value, int(balanced * len(data[0][i][target])), seed=seed)
                        #if len(neg[i]) > len(data[0][i][target]):
                        #    # NEW
                        #    amnt = math.ceil((2 if not balanced else balanced))
                        #    temp = datasets.generate_neg(target, data[0][i][target], amount=amnt, seed=seed)
                        #    temp = neg[i] + temp
                        #    temp = temp[:int(balanced * len(data[0][i][target]))]
                        #    neg[i] = temp
                    else:
                        neg[i] = datasets.get_neg(target, value)
                else:
                    value = data[0][i][target]
                    if balanced:
                        neg[i] = datasets.generate_neg(target, value, amount=(1 if not balanced else balanced), seed=seed)
                    else:
                        neg[i] = datasets.generate_all_neg(target, value)
        return [facts, pos, neg]

    def save():
        import time
        start = time.time()
        data = datasets.get_imdb_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'imdb'))
        with open('files/json/imdb.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_cora_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'cora'))
        with open('files/json/cora.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_uwcse_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'uwcse'))
        with open('files/json/uwcse.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_webkb2_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'webkb'))
        with open('files/json/webkb.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_nell_sports_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'nell_sports'))
        with open('files/json/nell_sports.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_nell_finances_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'nell_finances'))
        with open('files/json/nell_finances.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_yago2s_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'yago2s'))
        with open('files/json/yago2s.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_twitter_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'twitter'))
        with open('files/json/twitter.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_yeast_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'yeast'))
        with open('files/json/yeast.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_facebook_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'facebook'))
        with open('files/json/facebook.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_movielens_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'movielens'))
        with open('files/json/movielens.json', 'w') as outfile:
            json.dump(data, outfile)

        start = time.time()
        data = datasets.get_carcinogenesis_dataset()
        print('%s seconds generating %s' % (time.time() - start, 'carcinogenesis'))
        with open('files/json/carcinogenesis.json', 'w') as outfile:
            json.dump(data, outfile)

    '''
    workedunder(person,person)
    genre(person,genre)
    female(person)
    actor(person)
    director(person)
    movie(movie,person)
    genre(person,genre)
    genre(person,ascifi)
    genre(person,athriller)
    genre(person,adrama)
    genre(person,acrime)
    genre(person,acomedy)
    genre(person,amystery)
    genre(person,aromance)'''
    def get_imdb_dataset(acceptedPredicates=None):
        facts = []
        negatives = []
        i = -1
        with open(os.path.join(__location__, 'files/imdb.pl')) as f:
            for line in f:
                b = re.search('^begin\(model\([0-9\w]*\)\).$', line)
                m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
                n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)
                if b:
                    i += 1
                    facts.append({})
                    negatives.append({})
                if m:
                    relation = re.sub('[ _]', '', m.group(1))
                    entities = re.sub('[ _]', '', m.group(2)).split(',')
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[i]:
                            facts[i][relation] = []
                        facts[i][relation].append(entities)
                if n:
                    relation = re.sub('[ _]', '', n.group(1))
                    entities = re.sub('[ _]', '', n.group(2)).split(',')
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in negatives[i]:
                            negatives[i][relation] = []
                        negatives[i][relation].append(entities)
        return [facts, negatives]

    '''
    samebib(class,class)
    sameauthor(author,author)
    sametitle(title,title)
    samevenue(venue,venue)
    author(class,author)
    title(class,title)
    venue(class,venue)
    haswordauthor(author,word)
    harswordtitle(title,word)
    haswordvenue(venue,word)
    '''
    def get_cora_dataset(acceptedPredicates=None):
        facts = []
        negatives = []
        i = -1
        with open(os.path.join(__location__, 'files/coralearn.pl')) as f:
            for line in f:
                b = re.search('^begin\(model\([0-9\w]*\)\).$', line)
                n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)
                m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
                if b:
                    i += 1
                    facts.append({})
                    negatives.append({})
                if m:
                    relation = re.sub('[ _]', '', m.group(1))
                    entities = re.sub('[ _]', '', m.group(2)).split(',')
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[i]:
                            facts[i][relation] = []
                        facts[i][relation].append(entities)
#                if n:
#                    relation = re.sub('[ _]', '', n.group(1))
#                    entities = re.sub('[ _]', '', n.group(2)).split(',')
#                    if not acceptedPredicates or relation in acceptedPredicates:
#                        if relation not in negatives[i]:
#                            negatives[i][relation] = []
#                        negatives[i][relation].append(entities)
        return [facts, negatives]

    '''
    professor(person)
    student(person)
    advisedby(person,person)
    tempadvisedby(person,person)
    ta(course,person,quarter)
    hasposition(person,faculty)
    publication(title,person)
    inphase(person, pre_quals)
    taughtby(course, person, quarter)
    courselevel(course,#level)
    yearsinprogram(person,#year)
    projectmember(project, person)
    sameproject(project, project)
    samecourse(course, course)
    sameperson(person, person)'''
    def get_uwcse_dataset(acceptedPredicates=None):
        facts = []
        negatives = []
        fold = {}
        fold_i = 0
        i = 0
        with open(os.path.join(__location__, 'files/uwcselearn.pl')) as f:
            for line in f:
                n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)
                m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
                if m:
                    relation = re.sub('[ _]', '', m.group(1))
                    entities = re.sub('[ _]', '', m.group(2)).split(',')
                    if entities[0] not in fold:
                        fold[entities[0]] = fold_i
                        i = fold_i
                        facts.append({})
                        negatives.append({})
                        fold_i += 1
                    else:
                        i = fold[entities[0]]
                    entities = entities[1:]
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[i]:
                            facts[i][relation] = []
                        facts[i][relation].append(entities)
                if n:
                    relation = re.sub('[ _]', '', n.group(1))
                    entities = re.sub('[ _]', '', n.group(2)).split(',')
                    if entities[0] not in fold:
                        fold[entities[0]] = fold_i
                        i = fold_i
                        facts.append({})
                        negatives.append({})
                        fold_i += 1
                    else:
                        i = fold[entities[0]]
                    entities = entities[1:]
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in negatives[i]:
                            negatives[i][relation] = []
                        negatives[i][relation].append(entities)
        return [facts, negatives]

    '''
    coursepage(page)
    facultypage(page)
    studentpage(page)
    researchprojectpage(page)
    linkto(id,page,page)
    has(word,page)
    hasalphanumericword(id)
    allwordscapitalized(id)
    '''
    def get_webkb_dataset(acceptedPredicates=None):
        facts = []
        negatives = []
        pages = {}
        count = {'id' : 1}
        i = -1

        def getPageId(page):
            if page not in pages:
                pages[page] = 'page' + str(count['id'])
                count['id'] += 1
            return pages[page]

        def cleanEntity(entity):
            m = re.search('^(http|https|ftp|mail|file)\:', entity)
            if m:
                return getPageId(entity)
            else:
                return entity

        def getCleanEntities(entities):
            new_entities = list(entities)
            return [cleanEntity(entity) for entity in new_entities]

        with open(os.path.join(__location__, 'files/webkb.pl')) as f:
            for line in f:
                b = re.search('^begin\(model\([0-9\w]*\)\).$', line.lower())
                n = re.search('^neg\((\w+)\((.*)\)\).$', line.lower())
                m = re.search('^(\w+)\((.*)\).$', line.lower())
                if b:
                    i += 1
                    facts.append({})
                    negatives.append({})
                    continue
                if n:
                    relation = re.sub('[\']', '', n.group(1))
                    if relation not in ['output', 'input_cw', 'input', 'determination', 'begin', 'modeb', 'modeh', 'banned', 'fold', 'lookahead', 'bg', 'in']:
                        entities = re.sub('[\']', '', n.group(2)).split(',')
                        entities = getCleanEntities(entities)
                        if not acceptedPredicates or relation in acceptedPredicates:
                            if relation not in negatives[i]:
                                negatives[i][relation] = []
                            negatives[i][relation].append(entities)
                    continue
                if m:
                    relation = re.sub('[\']', '', m.group(1))
                    if relation not in ['output', 'input_cw', 'input', 'determination', 'begin', 'modeb', 'modeh', 'banned', 'fold', 'lookahead', 'bg', 'in']:
                        entities = re.sub('[\']', '', m.group(2)).split(',')
                        entities = getCleanEntities(entities)
                        if not acceptedPredicates or relation in acceptedPredicates:
                            if relation not in facts[i]:
                                facts[i][relation] = []
                            facts[i][relation].append(entities)
                    continue
        return [facts, negatives]

    '''
    coursepage(page)
    facultypage(page)
    studentpage(page)
    researchprojectpage(page)
    linkto(id,page,page)
    has(word,page)
    hasalphanumericword(id)
    allwordscapitalized(id)
    instructorsof(page,page)
    hasanchor(word,page)
    membersofproject(page,page)
    departmentof(page,page)
    pageclass(page,class)
    '''
    def get_webkb2_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z0-9]', '', value)
            return value

        facts = [{},{},{},{}]
        negatives = [{},{},{},{}]
        pages = {}
        count = {'id' : 1}
        i = -1

        def getPageId(page):
            if page not in pages:
                pages[page] = 'page' + str(count['id'])
                count['id'] += 1
            return pages[page]

        def cleanEntity(entity):
            m = re.search('(http|https|ftp|mail|file)', entity)
            if m:
                return getPageId(entity)
            else:
                return clearCharacters(entity)

        def getCleanEntities(entities):
            new_entities = list(entities)
            return [cleanEntity(entity) for entity in new_entities]

        classes = ['course', 'department', 'faculty', 'person', 'student', 'researchproject', 'staff']
        folds = ['cornell', 'texas', 'washington', 'wisconsin']
        files = ['background/anchor-words', 'background/common', 'background/page-classes',
                 'background/page-words', 'target/course', 'target/department-of', 'target/faculty', 'target/instructors-of',
                 'target/members-of-project', 'target/research.project', 'target/student']

        for i in range(len(folds)):
            fold = folds[i]
            for file in files:
                with open(os.path.join(__location__, 'files/webkb/' + file + '.' + fold + '.db')) as f:
                    for line in f:
                        n = re.search('^!(\w+)\((.*)\)$', line.lower())
                        m = re.search('^(\w+)\((.*)\)$', line.lower())
                        if n:
                            relation = n.group(1)
                            entities = n.group(2).split(',')
                            entities = getCleanEntities(entities)
                            if not acceptedPredicates or relation in acceptedPredicates:
                                if relation in classes:
                                    if 'pageclass' not in facts[i]:
                                        negatives[i]['pageclass'] = []
                                    entities += [relation]
                                    negatives[i]['pageclass'].append(entities)
                                else:
                                    if relation not in facts[i]:
                                        negatives[i][relation] = []
                                    negatives[i][relation].append(entities)
                            continue
                        if m:
                            relation = m.group(1)
                            entities = m.group(2).split(',')
                            entities = getCleanEntities(entities)
                            if not acceptedPredicates or relation in acceptedPredicates:
                                if relation in classes:
                                    if 'pageclass' not in facts[i]:
                                        facts[i]['pageclass'] = []
                                    entities += [relation]
                                    facts[i]['pageclass'].append(entities)
                                else:
                                    if relation not in facts[i]:
                                        facts[i][relation] = []
                                    facts[i][relation].append(entities)
                            continue
        return [facts, negatives]

    '''
    athleteledsportsteam(athlete,sportsteam)
    athleteplaysforteam(athlete,sportsteam)
    athleteplaysinleague(athlete,sportsleague)
    athleteplayssport(athlete,sport)
    teamalsoknownas(sportsteam,sportsteam)
    teamplaysagainstteam(sportsteam,sportsteam)
    teamplaysinleague(sportsteam,sportsleague)
    teamplayssport(sportsteam,sport)
    '''
    def get_nell_sports_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z]', '', value)
            return value

        facts = [{}]
        dataset = pd.read_csv(os.path.join(__location__, 'files/NELL.sports.08m.1070.small.csv'))
        for data in dataset.values:
            entity = clearCharacters((data[1].split(':'))[2])
            relation = clearCharacters((data[4].split(':'))[1])
            value = clearCharacters((data[5].split(':'))[2])

            if entity and relation and value:
                if not acceptedPredicates or relation in acceptedPredicates:
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    facts[0][relation].append([entity, value])
        return [facts, [{}]]

    '''
    countryhascompanyoffice(country,company)
    companyeconomicsector(company,sector)
    economicsectorcompany(sector,company)
    ceoeconomicsector(person,sector)
    companyceo(company,person)
    companyalsoknownas(company,company)
    cityhascompanyoffice(city,company)
    acquired(company,company)
    ceoof(person,company)
    bankbankincountry(company,company)
    bankboughtbank(company,company)
    bankchiefexecutiveceo(company,person)
    '''
    def get_nell_finances_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z]', '', value)
            return value

        companyceo = {}
        companyeconomicsector = {}
        bankchiefexecutiveceo = {}
        facts = [{}]
        dataset = pd.read_csv(os.path.join(__location__, 'files/NELL.finances.08m.1115.small.csv'))
        for data in dataset.values:
            entity = clearCharacters((data[1].split(':'))[2])
            relation = clearCharacters((data[4].split(':'))[1])
            value = clearCharacters((data[5].split(':'))[2])

            if entity and value:
                if relation == 'companyceo':
                    companyceo[entity] = value
                elif relation == 'companyeconomicsector':
                    companyeconomicsector[entity] = value
                elif relation == 'bankchiefexecutiveceo':
                    bankchiefexecutiveceo[entity] = value

            if entity and relation and value:
                if not acceptedPredicates or relation in acceptedPredicates:
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    facts[0][relation].append([entity, value])
        for key, value in companyceo.items():
            if key in companyeconomicsector:
                if 'ceoeconomicsector' not in facts[0]:
                    facts[0]['ceoeconomicsector'] = []
                facts[0]['ceoeconomicsector'].append([value, companyeconomicsector[key]])
        for key, value in bankchiefexecutiveceo.items():
            if key in companyeconomicsector:
                if 'ceoeconomicsector' not in facts[0]:
                    facts[0]['ceoeconomicsector'] = []
                facts[0]['ceoeconomicsector'].append([value, companyeconomicsector[key]])
        return [facts, [{}]]

    '''
    accounttype(account,+type)
    tweets(account,+word)
    follows(account,account)'''
    def get_twitter_dataset(acceptedPredicates=None):
        facts = [{},{}]
        for i in range(2):
            with open(os.path.join(__location__, 'files/twitter-fold' + str(i+1) + '.db')) as f:
                for line in f:
                    m = re.search('^([\w_]+)\(([\w, "_-]+)*\)$', line.lower())
                    if m:
                        relation = m.group(1)
                        entities = m.group(2)
                        entities = re.sub('[ _"-]', '', entities)
                        entities = entities.split(',')
                        if not acceptedPredicates or relation in acceptedPredicates:
                            if relation not in facts[i]:
                                facts[i][relation] = []
                            if relation == 'accounttype':
                                if 'typeaccount' not in facts[i]:
                                    facts[i]['typeaccount'] = []
                                facts[i]['typeaccount'].append(entities[::-1])
                            facts[i][relation].append(entities)
        return [facts, [{},{}]]

    '''
    location(protein,loc)
    interaction(protein,protein)
    proteinclass(protein,class)
    enzyme(protein,enz)
    function(protein,+fun)
    complex(protein,com)
    phenotype(protein,phe)'''
    def get_yeast_dataset(acceptedPredicates=None):
        facts = [{},{},{},{}]
        for i in range(4):
            with open(os.path.join(__location__, 'files/yeast-fold' + str(i+1) + '.db')) as f:
                for line in f:
                    m = re.search('^([\w_]+)\(([\w, "_-]+)*\)$', line.lower())
                    if m:
                        relation = m.group(1)
                        relation = re.sub('[_]', '', relation)
                        entities = m.group(2)
                        entities = re.sub('[ _"-]', '', entities)
                        entities = entities.split(',')
                        if not acceptedPredicates or relation in acceptedPredicates:
                            if relation not in facts[i]:
                                facts[i][relation] = []
                            if relation == 'proteinclass':
                                if 'classprotein' not in facts[i]:
                                    facts[i]['classprotein'] = []
                                facts[i]['classprotein'].append(entities[::-1])
                            facts[i][relation].append(entities)
        return [facts, [{},{},{},{}]]

    '''
    edge(person,person)
    middlename(person,middlename)
    lastname(person,lastname)
    educationtype(person,educationtype)
    workprojects(person,workprojects)
    educationyear(person,educationyear)
    educationwith(person,educationwith)
    location(person,location)
    workwith(person,workwith)
    workenddate(person,workenddate)
    languages(person,languages)
    religion(person,religion)
    political(person,political)
    workemployer(person,workemployer)
    hometown(person,hometown)
    educationconcentration(person,educationconcentration)
    workfrom(person,workfrom)
    workstartdate(person,workstartdate)
    worklocation(person,worklocation)
    educationclasses(person,educationclasses)
    workposition(person,workposition)
    firstname(person,firstname)
    birthday(person,birthday)
    educationschool(person,educationschool)
    name(person,name)
    gender(person,gender)
    educationdegree(person,educationdegree)
    locale(person,locale)'''
    def get_facebook_dataset(acceptedPredicates=None):
        folds_id = [0, 414, 686, 698, 3980] #[0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
        facts = []
        for fold in folds_id:
            fc = {}
            relation = 'edge'
            if not acceptedPredicates or relation in acceptedPredicates:
                if relation not in fc:
                    fc[relation] = []
                with open(os.path.join(__location__, 'files/facebook/' + str(fold) + '.edges')) as f:
                    for line in f:
                        data = line.split()
                        fc[relation].append(['person' + str(data[0]), 'person' + str(data[1])])
            featnames = []
            featpredicates = []
            with open(os.path.join(__location__, 'files/facebook/' + str(fold) + '.featnames')) as f:
                for line in f:
                    spl = line.split(';')
                    if len(spl) == 2:
                        pred = re.sub('[^a-z]', '', spl[0])
                        name = re.sub('[^0-9]', '', spl[1])
                    else:
                        pred = re.sub('[^a-z]', '', spl[0]) + re.sub('[^a-z]', '', spl[1])
                        pred = pred.replace('id', '')
                        name = re.sub('[^0-9]', '', spl[-1])
                    featnames.append(pred + name)
                    featpredicates.append(pred)
            with open(os.path.join(__location__, 'files/facebook/' + str(fold) + '.feat')) as f:
                for line in f:
                    spl = line.split()
                    person_id = 'person' + str(spl[0])
                    spl = spl[1:]
                    for i in range(len(spl)):
                        if int(spl[i]) == 1:
                            relation = featpredicates[i]
                            if relation not in fc:
                                fc[relation] = []
                            fc[relation].append([person_id, featnames[i]])
            facts.append(fc)
        return [facts, [{},{},{},{},{}]] # [facts, [{},{},{},{},{},{},{},{},{},{}]]

    '''
    actor(person)
    actorfemale(person)
    country(movie,country)
    director(person)
    genre(movie,genre)
    isenglish(movie)
    movie(movie,person)
    occupation(user,occupation)
    likes(user,movie)
    userfemale(user)
    age(user,age)
    '''
    def get_movielens_dataset(acceptedPredicates=None):
        import numpy as np
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z]', '', value)
            return value

        facts = [{}]
        dataset = pd.read_csv(os.path.join(__location__, 'files/movielens/movies.csv'), delimiter=',')
        movies = set()
        for data in dataset.values:
            movies.add('movie' + str(data[0]))
        movies = list(movies)
        from random import shuffle
        shuffle(movies)
        movies = movies[:603]
        for data in dataset.values:
            entity = 'movie' + str(data[0])
            if entity in movies:
                if data[2] == 'T':
                    relation = 'isenglish'
                    if entity and relation:
                        if not acceptedPredicates or relation in acceptedPredicates:
                            if relation not in facts[0]:
                                facts[0][relation] = []
                            facts[0][relation].append([entity])
            relation = 'country'
            value = data[3]
            if entity and relation:
                if not acceptedPredicates or relation in acceptedPredicates:
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    facts[0][relation].append([entity, value])
        person = set()
        dataset = pd.read_csv(os.path.join(__location__, 'files/movielens/movies2actors.csv'), delimiter=',')
        for data in dataset.values:
            entity = 'movie' + str(data[0])
            relation = 'movie'
            value = 'actor' + str(data[1])
            if entity in movies:
                person.add(value)
                if entity and relation:
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[0]:
                            facts[0][relation] = []
                        facts[0][relation].append([entity, value])
        dataset = pd.read_csv(os.path.join(__location__, 'files/movielens/movies2directors.csv'), delimiter=',')
        for data in dataset.values:
            entity = 'movie' + str(data[0])
            relation = 'movie'
            value = 'director' + str(data[1])
            if entity in movies:
                person.add(value)
                if entity and relation:
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[0]:
                            facts[0][relation] = []
                        facts[0][relation].append([entity, value])
                relation = 'genre'
                value = data[2]
                if entity and relation:
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[0]:
                            facts[0][relation] = []
                        facts[0][relation].append([entity, value])
        dataset = pd.read_csv(os.path.join(__location__, 'files/movielens/actors.csv'), delimiter=',')
        for data in dataset.values:
            entity = 'actor' + str(data[0])
            if entity in person:
                if data[1] == 'F':
                    relation = 'actorfemale'
                    if entity and relation:
                        if not acceptedPredicates or relation in acceptedPredicates:
                            if relation not in facts[0]:
                                facts[0][relation] = []
                            facts[0][relation].append([entity])
                relation = 'actor'
                if entity and relation:
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[0]:
                            facts[0][relation] = []
                        facts[0][relation].append([entity])
        dataset = pd.read_csv(os.path.join(__location__, 'files/movielens/directors.csv'), delimiter=',')
        for data in dataset.values:
            entity = 'director' + str(data[0])
            relation = 'director'
            if entity in person:
                if entity and relation:
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[0]:
                            facts[0][relation] = []
                        facts[0][relation].append([entity])
        dataset = pd.read_csv(os.path.join(__location__, 'files/movielens/u2base.csv'), delimiter=',')
        rates = {}
        users = set()
        for data in dataset.values:
            users.add('user' + str(data[0]))
        users = list(users)
        shuffle(users)
        users = users[:100]
        for data in dataset.values:
            entity = 'user' + str(data[0])
            value = int(data[2])
            if entity not in rates:
                rates[entity] = []
            rates[entity].append(value)
        for data in dataset.values:
            entity = 'user' + str(data[0])
            relation = 'likes'
            value = 'movie' + str(data[1])
            if entity in users and value in movies:
                if entity and relation:
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[0]:
                            facts[0][relation] = []
                        if int(data[2]) > (np.array(rates[entity])).mean():
                            facts[0][relation].append([entity, value])
        dataset = pd.read_csv(os.path.join(__location__, 'files/movielens/users.csv'), delimiter=',')
        for data in dataset.values:
            entity = 'user' + str(data[0])
            if entity in users:
                if data[2] == 'F':
                    relation = 'userfemale'
                    if entity and relation:
                        if not acceptedPredicates or relation in acceptedPredicates:
                            if relation not in facts[0]:
                                facts[0][relation] = []
                            facts[0][relation].append([entity])
                relation = 'occupation'
                value = 'occupation' + str(data[3])
                if entity and relation:
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[0]:
                            facts[0][relation] = []
                        facts[0][relation].append([entity, value])
                relation = 'age'
                value = 'age' + str(data[1])
                if entity and relation:
                    if not acceptedPredicates or relation in acceptedPredicates:
                        if relation not in facts[0]:
                            facts[0][relation] = []
                        facts[0][relation].append([entity, value])
        return [facts, [{}]]

    '''
    positive(drug)
    atomtype(atom, type)
    charge(atom, charge)
    drug(atom, drug)
    name(atom, name)
    sbond1atom1(drug, atom)
    sbond1atom2(drug, atom)
    sbond2atom1(drug, atom)
    sbond2atom2(drug, atom)
    sbond3atom1(drug, atom)
    sbond3atom2(drug, atom)
    sbond7atom1(drug, atom)
    sbond7atom2(drug, atom)
    '''
    def get_carcinogenesis_dataset(acceptedPredicates=None):
        import numpy as np
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z_0-9]', '', value)
            value = value.replace('_', 'u')
            return value

        facts = [{}]
        negatives = [{}]
        dataset = pd.read_csv(os.path.join(__location__, 'files/carcinogenesis/atom.csv'), delimiter=';', header=0)
        charge = {}
        for data in dataset.values:
            entity = clearCharacters(str(data[0]))
            relation = 'drug'
            value = clearCharacters(str(data[1]))
            if not acceptedPredicates or relation in acceptedPredicates:
                if relation not in facts[0]:
                    facts[0][relation] = []
                facts[0][relation].append([entity, value])
            relation = 'atomtype'
            value = clearCharacters('type' + str(data[2]))
            if not acceptedPredicates or relation in acceptedPredicates:
                if relation not in facts[0]:
                    facts[0][relation] = []
                facts[0][relation].append([entity, value])
            relation = 'charge'
            value = clearCharacters(str(data[3]))
            if value not in charge:
                charge[value] = 'charge' + str(len(charge) + 1)
            value = charge[value]
            if not acceptedPredicates or relation in acceptedPredicates:
                if relation not in facts[0]:
                    facts[0][relation] = []
                facts[0][relation].append([entity, value])
            relation = 'name'
            value = clearCharacters('name' + str(data[4]))
            if not acceptedPredicates or relation in acceptedPredicates:
                if relation not in facts[0]:
                    facts[0][relation] = []
                facts[0][relation].append([entity, value])
        for sbond in ['1', '2', '3', '7']:
            dataset = pd.read_csv(os.path.join(__location__, 'files/carcinogenesis/sbond_' + sbond + '.csv'), delimiter=';', header=0)
            for data in dataset.values:
                entity = clearCharacters(str(data[1]))
                relation = 'sbond' + sbond + 'atom1'
                value = clearCharacters(str(data[2]))
                if not acceptedPredicates or relation in acceptedPredicates:
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    facts[0][relation].append([entity, value])
                entity = clearCharacters(str(data[1]))
                relation = 'sbond' + sbond + 'atom_2'
                value = clearCharacters(str(data[3]))
                if not acceptedPredicates or relation in acceptedPredicates:
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    facts[0][relation].append([entity, value])
        dataset = pd.read_csv(os.path.join(__location__, 'files/carcinogenesis/canc.csv'), delimiter=';', header=0)
        for data in dataset.values:
            entity = clearCharacters(str(data[0]))
            relation = 'positive'
            value = clearCharacters(str(data[1]))
            if value == '1':
                if relation not in facts[0]:
                    facts[0][relation] = []
                facts[0][relation].append([entity])
            else:
                if relation not in negatives[0]:
                    negatives[0][relation] = []
                negatives[0][relation].append([entity])
        return [facts, negatives]

    '''
    playsfor(person,team),
    hascurrency(place,currency),
    hascapital(place,place),
    hasacademicadvisor(person,person),
    haswonprize(person,prize),
    participatedin(place,event),
    owns(institution,institution),
    isinterestedin(person,concept),
    livesin(person,place),
    happenedin(event,place),
    holdspoliticalposition(person,politicalposition),
    diedin(person,place),
    actedin(person,media),
    iscitizenof(person,place),
    worksat(person,institution),
    directed(person,media),
    dealswith(place,place),
    wasbornin(person,place),
    created(person,media),
    isleaderof(person,place),
    haschild(person,person),
    ismarriedto(person,person),
    imports(person,material),
    hasmusicalrole(person,musicalrole),
    influences(person,person),
    isaffiliatedto(person,team),
    isknownfor(person,theory),
    ispoliticianof(person,place),
    graduatedfrom(person,institution),
    exports(place,material),
    edited(person,media),
    wrotemusicfor(person,media).
    '''
    def get_yago2s_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z0-9]', '', value)
            return value
        '''<Paul_Redford>   <created>   <The_Portland_Trip> .'''
        facts = [{}]
        dataset = pd.read_csv(os.path.join(__location__, 'files/yago2s.tsv'), sep='\t').drop_duplicates()
        for data in dataset.values:
            entity   = clearCharacters(data[0].split(',')[0])
            relation = clearCharacters(data[1])
            value    = clearCharacters(str(data[2].split(',')[0]))

            if entity and relation and value:
                if not acceptedPredicates or relation in acceptedPredicates:
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    facts[0][relation].append([entity, value])
        return [facts, [{}]]

    '''
    cites(paper,paper),
    gene(paper,gene),
    journal(paper,journal),
    author(paper,author),
    chem(paper,chemical),
    aff(paper,institute),
    aspect(paper,gene,R)
    '''
    def get_yeast2_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z0-9]', '', value)
            return value
        '''Author(65102,SwietliÅska_Z Zaborowska_D Zuk_J)'''
        '''Aff(1408783,Centre_d'Etudes_de_Saclay)'''
        facts = [{}]

        #Dicts for validation
        authors, cites, papers, journals, chem, genes, aspect = {}, {}, {}, {}, {}, {}, {}

        with open(os.path.join(__location__, 'files/yeast2/alchemy/Abstract.db'), encoding="utf8") as f:
            for line in f:
                if 'FAuthor' in line or 'LAuthor' in line or 'DmHead' in line or 'QmHead' in line:
                    continue
                if 'Journal' in line or 'Aff' in line:
                    relation = clearCharacters(line.split('(')[0])
                    entity   = clearCharacters(line.split('(')[1].split(',')[0])
                    value    = clearCharacters(line.split(',')[1])
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    facts[0][relation].append([entity, value])

                    if entity not in papers:
                        papers[entity] = 0

                    if value not in journals:
                        journals[value] = 0

                elif 'Author' in line or 'Chem' in line:
                    relation = clearCharacters(line.split('(')[0])
                    entity   = clearCharacters(line.split('(')[1].split(',')[0])
                    values   = line.split(',')[1].split()
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    for value in values:
                        value = clearCharacters(value)
                        facts[0][relation].append([entity, value])

                        if 'Chem' in line and value not in chem:
                            chem[value] = 0

                        if 'Author' in line and value not in authors:
                            authors[value] = 0

        with open(os.path.join(__location__, 'files/yeast2/alchemy/GeneCitation.db'), encoding="utf8") as f:
            for line in f:
                relation = clearCharacters(line.split('(')[0])
                entity_1 = clearCharacters(line.split('(')[1].split(',')[0])

                if 'Aspect' in line:
                    entity_2 = clearCharacters(line.split('(')[1].split(',')[1])
                    values   = line.split(',')[2].split()
                else:
                    values   = line.split(',')[1].split()

                if relation not in facts[0]:
                    facts[0][relation] = []
                for value in values:
                    value = clearCharacters(value)

                    if 'Aspect' in line:
                        facts[0][relation].append([entity_1, entity_2, value])

                    if 'Gene' in line:
                        facts[0][relation].append([entity_1, value])

                        if value not in genes:
                            genes[value] = 0

                    if entity not in papers:
                        papers[entity] = 0

        with open(os.path.join(__location__, 'files/yeast2/alchemy/Ref.db'), encoding="utf8") as f:
            for line in f:
                relation = clearCharacters(line.split('(')[0])
                entity   = clearCharacters(line.split('(')[1].split(',')[0])
                values   = line.split(',')[1].split()
                if relation not in facts[0]:
                    facts[0][relation] = []
                for value in values:
                    value = clearCharacters(value)
                    facts[0][relation].append([entity, value])
        with open(os.path.join(__location__, 'files/yeast2/alchemy/RefSGD.db'), encoding="utf8") as f:
            for line in f:
                relation = clearCharacters(line.split('(')[0])
                entity   = clearCharacters(line.split('(')[1].split(',')[0])
                values   = line.split(',')[1].split()
                if relation not in facts[0]:
                    facts[0][relation] = []
                for value in values:
                    value = clearCharacters(value)
                    facts[0][relation].append([entity, value])

        citations = []
        for tupla in facts[0]['cites']:
            if tupla[1] in papers:
                citations.append([tupla[0], tupla[1]])
        facts[0]['cites'] = citations.copy()

        print('Total of authors {}'.format(len(authors)))
        print('Total of papers {}'.format(len(papers)))
        print('Total of genes {}'.format(len(genes)))
        print('Total of chemicals {}'.format(len(chem)))
        print('Total of journals {}'.format(len(journals)))
        # 159k papers
        # 14k chemicals
        # 71k authors
        # 1k journals
        # 1.9k affiliations (selected from 6k affiliations)
        # 5.6k genes

        print('{} aff interactions'.format(len(facts[0]['aff'])))
        print('{} citations'.format(len(facts[0]['cites'])))

        return [facts, [{}]]

    '''
    journal(paper,journal),
    author(paper,author),
    cites(paper,paper),
    gene(paper,gene),
    aspect(paper,gene,R),
    gp(gene,protein),
    genetic(gene,gene),
    physical(gene,gene)
    '''
    def get_fly_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = unidecode.unidecode(value)
            value = re.sub('[^a-z0-9]', '', value)
            return value
        facts = [{}]

        #Dicts for validation
        authors, cites, proteins, genes, aspect, gp, papers, journals = {}, {}, {}, {}, {}, {}, {}, {}

        '''Author(65102,SwietliAska_Z Zaborowska_D Zuk_J)'''
        '''Journal(18791224,Genetics)'''
        with open(os.path.join(__location__, 'files/fly/alchemy/papers.db'), encoding="utf8") as f:
            for line in f:
                if 'FAuthor' in line:
                    continue
                if 'Journal' in line:
                    relation = clearCharacters(line.split('(')[0])
                    entity   = clearCharacters(line.split('(')[1].split(',')[0])
                    value    = clearCharacters(line.split(',')[1])
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    facts[0][relation].append([entity, value])

                    if entity not in papers:
                        papers[entity] = 0

                    if value not in journals:
                        journals[value] = 0

                elif 'Author' in line or 'Cites' in line or 'Protein' in line:
                    relation = clearCharacters(line.split('(')[0])
                    entity   = clearCharacters(line.split('(')[1].split(',')[0])
                    values   = line.split(',')[1].split()
                    if relation not in facts[0]:
                        facts[0][relation] = []
                    for value in values:
                        value = clearCharacters(value)
                        facts[0][relation].append([entity, value])

                        if entity not in papers:
                            papers[entity] = 0
                        if('Author' in line and value not in authors):
                            authors[value] = 0
                        if('Protein' in line and value not in proteins):
                            proteins[value] = 0
        citations = []
        for tupla in facts[0]['cites']:
            if tupla[1] in papers:
                citations.append([tupla[0], tupla[1]])
        facts[0]['cites'] = citations.copy()

        '''Aspect(55,TDH3,FuPr SLC)'''
        '''Gene(55,TPI1 TDH3)'''
        with open(os.path.join(__location__, 'files/fly/alchemy/GeneCitation.db'), encoding="utf8") as f:
            for line in f:
                relation = clearCharacters(line.split('(')[0])
                entity_1 = clearCharacters(line.split('(')[1].split(',')[0])

                if 'Aspect' in line:
                    entity_2 = clearCharacters(line.split('(')[1].split(',')[1])
                    values   = line.split(',')[2].split()
                else:
                    values   = line.split(',')[1].split()

                if relation not in facts[0]:
                    facts[0][relation] = []
                for value in values:
                    value = clearCharacters(value)

                    if 'Aspect' in line:
                        facts[0][relation].append([entity_1, entity_2, value])

                    if 'Gene' in line:
                        facts[0][relation].append([entity_1, value])

                        if value not in genes:
                            genes[value] = 0

                    if entity not in papers:
                        papers[entity] = 0

        '''GP(2L52.1,A4F336_CAEEL)'''
        with open(os.path.join(__location__, 'files/fly/alchemy/gene2Protein.db'), encoding="utf8") as f:
            for line in f:
                relation = clearCharacters(line.split('(')[0])
                entity   = clearCharacters(line.split('(')[1].split(',')[0])
                values   = line.split(',')[1].split()
                if relation not in facts[0]:
                    facts[0][relation] = []
                for value in values:
                    value = clearCharacters(value.split('_')[0])
                    facts[0][relation].append([entity, value])

        '''physical(AC3.4,AC3.3)'''
        '''genetic(AC7.2,C54D1.6 ZK792.6)'''
        with open(os.path.join(__location__, 'files/fly/alchemy/geneInter.db'), encoding="utf8") as f:
            for line in f:
                relation = clearCharacters(line.split('(')[0])
                entity   = clearCharacters(line.split('(')[1].split(',')[0])
                values   = line.split(',')[1].split()
                if relation not in facts[0]:
                    facts[0][relation] = []
                for value in values:
                    value = clearCharacters(value)
                    facts[0][relation].append([entity, value])

        print('Total of authors {}'.format(len(authors)))
        print('Total of papers {}'.format(len(papers)))
        print('Total of genes {}'.format(len(genes)))
        print('Total of proteins {}'.format(len(proteins)))
        print('Total of journals {}'.format(len(journals)))
        # 39,037 papers in PMC, 385,699 in total if cited papers are included
        # 102,472  authors.
        # 244,014  genes in flymine.
        # 340,039 proteins in flymine.
        # 376 journals

        # 679,903 Citation relations.
        # 550,458 physical/genetic interaction relations from genes to other genes.
        print('{} physical interactions'.format(len(facts[0]['physical'])))
        print('{} genetic interactions'.format(len(facts[0]['genetic'])))
        print('{} citations'.format(len(facts[0]['cites'])))

        return [facts, [{}]]



#import time
#start = time.time()
#data = datasets.get_webkb2_dataset()
#print(time.time() - start)
#
#import json
#with open('files/json/webkb.json', 'w') as outfile:
#    json.dump(data, outfile)

#import time
#start = time.time()
#data = datasets.get_json_dataset('webkb')
#print(time.time() - start)
#
#start = time.time()
#data2 = datasets.target('advisedby', data)
#print(time.time() - start)

#start = time.time()
#data = datasets.load('webkb', ['coursepage(page).',
#    'facultypage(page).',
#    'studentpage(page).',
#    'researchprojectpage(page).',
#    'linkto(id,page,page).',
#    'has(word,page).',
#    'hasalphanumericword(id).',
#    'allwordscapitalized(id).',
#    'departmentof(page,page).',
#    'pageclass(page,class).'], target='pageclass')
#print(time.time() - start)

#import time
#start = time.time()
#data2 = datasets.load('uwcse', ['professor(person).',
#    'student(person).',
#    'advisedby(person,person)'
#    'tempadvisedby(person,person).',
#    'hasposition(person,faculty).',
#    'publication(title,person).',
#    'inphase(person, pre_quals).',
#    'courselevel(course,#level).',
#    'yearsinprogram(person,#year).'], target='advisedby')
#print(time.time() - start)


#import time
#start = time.time()
#data = datasets.get_yago2s_dataset()
#print(time.time() - start)

#import json
#with open('files/json/yago2s.json', 'w') as outfile:
#    json.dump(data,outfile)

#import time
#start = time.time()
#data = datasets.get_yeast2_dataset()
#print(time.time() - start)

#import json
#with open('files/json/yeast2.json', 'w') as outfile:
#    json.dump(data,outfile)

#import time
#start = time.time()
#data = datasets.get_fly_dataset()
#print(time.time() - start)

#import json
#with open('files/json/fly.json', 'w') as outfile:
#    json.dump(data,outfile)