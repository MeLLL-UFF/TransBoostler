
revision_theory = True

setups = [
        {'model': 'fasttext', 'similarity_metric': 'softcosine', 'revision_theory': revision_theory},
        {'model': 'fasttext', 'similarity_metric': 'wmd', 'revision_theory': revision_theory},
        {'model': 'fasttext', 'similarity_metric': 'relax-wmd', 'revision_theory': revision_theory},
        {'model': 'fasttext', 'similarity_metric': 'euclidean', 'revision_theory': revision_theory},
        {'model': 'fasttext', 'similarity_metric': 'cosine', 'revision_theory': revision_theory},
        
        {'model': 'word2vec', 'similarity_metric': 'softcosine', 'revision_theory': revision_theory},
        {'model': 'word2vec', 'similarity_metric': 'wmd', 'revision_theory': revision_theory},
        {'model': 'word2vec', 'similarity_metric': 'relax-wmd', 'revision_theory': revision_theory},
        {'model': 'word2vec', 'similarity_metric': 'euclidean', 'revision_theory': revision_theory},
        {'model': 'word2vec', 'similarity_metric': 'cosine', 'revision_theory': revision_theory},
        ]

experiments = [
            {'id': '1', 'source':'imdb', 'target':'uwcse', 'predicate':'workedunder', 'to_predicate':'advisedby', 'arity': 2},
            {'id': '2', 'source':'uwcse', 'target':'imdb', 'predicate':'advisedby', 'to_predicate':'workedunder', 'arity': 2},
            {'id': '3', 'source':'imdb', 'target':'cora', 'predicate':'workedunder', 'to_predicate':'samevenue', 'arity': 2},
            {'id': '4', 'source':'cora', 'target':'imdb', 'predicate':'samevenue', 'to_predicate':'workedunder', 'arity': 2},
            ##{'id': '5', 'source':'uwcse', 'target':'cora', 'predicate':'advisedby', 'to_predicate':'samevenue', 'arity': 2},
            ##{'id': '6', 'source':'cora', 'target':'uwcse', 'predicate':'samevenue', 'to_predicate':'advisedby', 'arity': 2},
            {'id': '7', 'source':'yeast', 'target':'twitter', 'predicate':'proteinclass', 'to_predicate':'accounttype', 'arity': 2},
            {'id': '8', 'source':'twitter', 'target':'yeast', 'predicate':'accounttype', 'to_predicate':'proteinclass', 'arity': 2},
            {'id': '9', 'source':'nell_sports', 'target':'nell_finances', 'predicate':'teamplayssport', 'to_predicate':'companyeconomicsector', 'arity': 2},
            {'id': '10', 'source':'nell_finances', 'target':'nell_sports', 'predicate':'companyeconomicsector', 'to_predicate':'teamplayssport', 'arity': 2},
            #{'id': '11', 'source':'yeast', 'target':'webkb', 'predicate':'proteinclass', 'to_predicate':'departmentof', 'arity':2},
            #{'id': '12', 'source':'webkb', 'target':'yeast', 'predicate':'departmentof', 'to_predicate':'proteinclass', 'arity':2},
            #{'id': '13', 'source': 'yago2s', 'target': 'yeast', 'predicate': 'wasbornin', 'to_predicate': 'proteinclass', 'arity': 2},
            #{'id': '14', 'source': 'yeast', 'target': 'yago2s', 'predicate': 'proteinclass', 'to_predicate': 'wasbornin', 'arity': 2},
            #{'id': '15', 'source': 'yeast', 'target': 'yeast2', 'predicate': 'proteinclass', 'to_predicate': 'gene', 'arity': 2},
            #{'id': '16', 'source': 'yeast', 'target': 'fly', 'predicate': 'proteinclass', 'to_predicate': 'gene', 'arity': 2},
            #{'id': '48', 'source':'twitter', 'target':'facebook', 'predicate':'follows', 'to_predicate':'edge', 'arity': 2},
            #{'id': '49', 'source':'imdb', 'target':'facebook', 'predicate':'workedunder', 'to_predicate':'edge','arity': 2},
            #{'id': '50', 'source':'uwcse', 'target':'facebook', 'predicate':'advisedby', 'to_predicate':'edge', 'arity': 2},
            ]
            
bk = {
      'imdb': ['workedunder(+person,+person).',
              'workedunder(+person,-person).',
              'workedunder(-person,+person).',
              #'recursion_workedunder(+person,`person).',
              #'recursion_workedunder(`person,+person).',
              'female(+person).',
              'actor(+person).',
              'director(+person).',
              'movie(+movie,+person).',
              'movie(+movie,-person).',
              'movie(-movie,+person).',
              'genre(+person,+genre).'],
      'uwcse': ['professor(+person).',
        'student(+person).',
        'advisedby(+person,+person).',
        'advisedby(+person,-person).',
        'advisedby(-person,+person).',
        #'recursion_advisedby(`person,+person).',
        #'recursion_advisedby(+person,`person).',
        'tempadvisedby(+person,+person).',
        'tempadvisedby(+person,-person).',
        'tempadvisedby(-person,+person).',
        'ta(+course,+person,+quarter).',
        'ta(-course,-person,+quarter).',
        'ta(+course,-person,-quarter).',
        'ta(-course,+person,-quarter).',
        'hasposition(+person,+faculty).',
        'hasposition(+person,-faculty).',
        'hasposition(-person,+faculty).',
        'publication(+title,+person).',
        'publication(+title,-person).',
        'publication(-title,+person).',
        'inphase(+person,+prequals).',
        'inphase(+person,-prequals).',
        'inphase(-person,+prequals).',
        'courselevel(+course,+level).',
        'courselevel(+course,-level).',
        'courselevel(-course,+level).',
        'yearsinprogram(+person,+year).',
        'yearsinprogram(-person,+year).',
        'yearsinprogram(+person,-year).',
        'projectmember(+project,+person).',
        'projectmember(+project,-person).',
        'projectmember(-project,+person).',
        'sameproject(+project,+project).',
        'sameproject(+project,-project).',
        'sameproject(-project,+project).',
        'samecourse(+course,+course).',
        'samecourse(+course,-course).',
        'samecourse(-course,+course).',
        'sameperson(+person,+person).',
        'sameperson(+person,-person).',
        'sameperson(-person,+person).',],
      'cora': ['sameauthor(+author,+author).',
              'sameauthor(+author,-author).',
              'sameauthor(-author,+author).',
              'samebib(+class,+class).',
              'samebib(+class,-class).',
              'samebib(-class,+class).',
              'sametitle(+title,+title).',
              'sametitle(+title,-title).',
              'sametitle(-title,+title).',
              'samevenue(+venue,+venue).',
              'samevenue(+venue,-venue).',
              'samevenue(-venue,+venue).',
              #'recursion_samevenue(+venue,`venue).',
              #'recursion_samevenue(`venue,+venue).',
              'author(+class,+author).',
              'author(+class,-author).',
              'author(-class,+author).',
              'title(+class,+title).',
              'title(+class,-title).',
              'title(-class,+title).',
              'venue(+class,+venue).',
              'venue(+class,-venue).',
              'venue(-class,+venue).',
              'haswordauthor(+author,+word).',
              'haswordauthor(+author,-word).',
              'haswordauthor(-author,+word).',
              'haswordtitle(+title,+word).',
              'haswordtitle(+title,-word).',
              'haswordtitle(-title,+word).',
              'haswordvenue(+venue,+word).',
              'haswordvenue(+venue,-word).',
              'haswordvenue(-venue,+word).'],
      'webkb': ['coursepage(+page).',
                'facultypage(+page).',
                'studentpage(+page).',
                'researchprojectpage(+page).',
                'linkto(+id,+page,+page).',
                'linkto(+id,-page,-page).',
                'linkto(-id,-page,+page).',
                'linkto(-id,+page,-page).',
                'has(+word,+page).',
                'has(+word,-page).',
                'has(-word,+page).',
                'hasalphanumericword(+id).',
                'allwordscapitalized(+id).',
                'instructorsof(+page,+page).',
                'instructorsof(+page,-page).',
                'instructorsof(-page,+page).',
                'hasanchor(+word,+page).',
                'hasanchor(+word,-page).',
                'hasanchor(-word,+page).',
                'membersofproject(+page,+page).',
                'membersofproject(+page,-page).',
                'membersofproject(-page,+page).',
                'departmentof(+page,+page).',
                'departmentof(+page,-page).',
                'departmentof(-page,+page).',
                'pageclass(+page,+class).',
                'pageclass(+page,-class).',
                'pageclass(-page,+class).'],
      'twitter': ['accounttype(+account,+type).',
                  'accounttype(+account,-type).',
                  'accounttype(-account,+type).',
                  #'typeaccount(+type,`account).',
                  #'typeaccount(`type,+account).',
                  'tweets(+account,+word).',
                  'tweets(+account,-word).',
                  'tweets(-account,+word).',
                  'follows(+account,+account).',
                  'follows(+account,-account).',
                  'follows(-account,+account).',
                  'recursion_accounttype(+account,`type).',
                  'recursion_accounttype(`account,+type).',],
      'yeast': ['location(+protein,+loc).',
                'location(+protein,-loc).',
                'location(-protein,+loc).',
                'interaction(+protein,+protein).',
                'interaction(+protein,-protein).',
                'interaction(-protein,+protein).',
                'proteinclass(+protein,+class).',
                'proteinclass(+protein,-class).',
                'proteinclass(-protein,+class).',
                #'classprotein(+class,`protein).',
                #'classprotein(`class,+protein).',
                'enzyme(+protein,+enz).',
                'enzyme(+protein,-enz).',
                'enzyme(-protein,+enz).',
                'function(+protein,+fun).',
                'function(+protein,-fun).',
                'function(-protein,+fun).',
                'complex(+protein,+com).',
                'complex(+protein,-com).',
                'complex(-protein,+com).',
                'phenotype(+protein,+phe).',
                'phenotype(+protein,-phe).',
                'phenotype(-protein,+phe).',
                'recursion_proteinclass(+protein,`class).',
                'recursion_proteinclass(`protein,+class).'],
      'nell_sports': ['athleteledsportsteam(+athlete,+sportsteam).',
              'athleteledsportsteam(+athlete,-sportsteam).',
              'athleteledsportsteam(-athlete,+sportsteam).',
              'athleteplaysforteam(+athlete,+sportsteam).',
              'athleteplaysforteam(+athlete,-sportsteam).',
              'athleteplaysforteam(-athlete,+sportsteam).',
              'athleteplaysinleague(+athlete,+sportsleague).',
              'athleteplaysinleague(+athlete,-sportsleague).',
              'athleteplaysinleague(-athlete,+sportsleague).',
              'athleteplayssport(+athlete,+sport).',
              'athleteplayssport(+athlete,-sport).',
              'athleteplayssport(-athlete,+sport).',
              'teamalsoknownas(+sportsteam,+sportsteam).',
              'teamalsoknownas(+sportsteam,-sportsteam).',
              'teamalsoknownas(-sportsteam,+sportsteam).',
              'teamplaysagainstteam(+sportsteam,+sportsteam).',
              'teamplaysagainstteam(+sportsteam,-sportsteam).',
              'teamplaysagainstteam(-sportsteam,+sportsteam).',
              'teamplaysinleague(+sportsteam,+sportsleague).',
              'teamplaysinleague(+sportsteam,-sportsleague).',
              'teamplaysinleague(-sportsteam,+sportsleague).',
              'teamplayssport(+sportsteam,+sport).',
              'teamplayssport(+sportsteam,-sport).',
              'teamplayssport(-sportsteam,+sport).',
              'teamplayssport(+sportsteam,`sport).',
              'teamplayssport(`sportsteam,+sport).',
              'recursion_teamplayssport(`sportsteam,+sport).',
              'recursion_teamplayssport(+sportsteam,`sport).'],
      'nell_finances': ['countryhascompanyoffice(+country,+company).',
                        'countryhascompanyoffice(+country,-company).',
                        'countryhascompanyoffice(-country,+company).',
                        'companyeconomicsector(+company,+sector).',
                        'companyeconomicsector(+company,-sector).',
                        'companyeconomicsector(-company,+sector).',
                        'economicsectorcompany(+sector,`company).',
                        'economicsectorcompany(`sector,+company).',
                        'recursion_economicsectorcompany(+sector,`company).',
                        'recursion_economicsectorcompany(`sector,+company).',
                        #'economicsectorcompany(+sector,+company).',
                        #'economicsectorcompany(+sector,-company).',
                        #'economicsectorcompany(-sector,+company).',
                        #'ceoeconomicsector(+person,+sector).',
                        #'ceoeconomicsector(+person,-sector).',
                        #'ceoeconomicsector(-person,+sector).',
                        'companyceo(+company,+person).',
                        'companyceo(+company,-person).',
                        'companyceo(-company,+person).',
                        'companyalsoknownas(+company,+company).',
                        'companyalsoknownas(+company,-company).',
                        'companyalsoknownas(-company,+company).',
                        'cityhascompanyoffice(+city,+company).',
                        'cityhascompanyoffice(+city,-company).',
                        'cityhascompanyoffice(-city,+company).',
                        'acquired(+company,+company).',
                        'acquired(+company,-company).',
                        'acquired(-company,+company).',
                        #'ceoof(+person,+company).',
                        #'ceoof(+person,-company).',
                        #'ceoof(-person,+company).',
                        'bankbankincountry(+person,+country).',
                        'bankbankincountry(+person,-country).',
                        'bankbankincountry(-person,+country).',
                        'bankboughtbank(+company,+company).',
                        'bankboughtbank(+company,-company).',
                        'bankboughtbank(-company,+company).',
                        'bankchiefexecutiveceo(+company,+person).',
                        'bankchiefexecutiveceo(+company,-person).',
                        'bankchiefexecutiveceo(-company,+person).'],              
      'yago2s': ['playsfor(+person,+team).',
    'playsfor(+person,-team).',
    'playsfor(-person,+team).',
    'hascurrency(+place,+currency).',
    'hascurrency(+place,-currency).',
    'hascurrency(-place,+currency).',
    'hascapital(+place,+place).',
    'hascapital(+place,-place).',
    'hascapital(-place,+place).',
    'hasacademicadvisor(+person,+person).',
    'hasacademicadvisor(+person,-person).',
    'hasacademicadvisor(-person,+person).',
    'haswonprize(+person,+prize).',
    'haswonprize(+person,-prize).',
    'haswonprize(-person,+prize).',
    'participatedin(+place,+event).',
    'participatedin(+place,-event).',
    'participatedin(-place,+event).',
    'owns(+institution,+institution).',
    'owns(+institution,-institution).',
    'owns(-institution,+institution).',
    'isinterestedin(+person,+concept).',
    'isinterestedin(+person,-concept).',
    'isinterestedin(-person,+concept).',
    'livesin(+person,+place).',
    'livesin(+person,-place).',
    'livesin(-person,+place).',
    'happenedin(+event,+place).',
    'happenedin(+event,-place).',
    'happenedin(-event,+place).',
    'holdspoliticalposition(+person,+politicalposition).',
    'holdspoliticalposition(+person,-politicalposition).',
    'holdspoliticalposition(-person,+politicalposition).',
    'diedin(+person,+place).',
    'diedin(+person,-place).',
    'diedin(-person,+place).',
    'actedin(+person,+media).',
    'actedin(+person,-media).',
    'actedin(-person,+media).',
    'iscitizenof(+person,+place).',
    'iscitizenof(+person,-place).',
    'iscitizenof(-person,+place).',
    'worksat(+person,+institution).',
    'worksat(+person,-institution).',
    'worksat(-person,+institution).',
    'directed(+person,+media).',
    'directed(+person,-media).',
    'directed(-person,+media).',
    'dealswith(+place,+place).',
    'dealswith(+place,-place).',
    'dealswith(-place,+place).',
    'wasbornin(+person,+place).',
    'wasbornin(+person,-place).',
    'wasbornin(-person,+place).',
    'created(+person,+media).',
    'created(+person,-media).',
    'created(-person,+media).',
    'isleaderof(+person,+place).',
    'isleaderof(+person,-place).',
    'isleaderof(-person,+place).',
    'haschild(+person,+person).',
    'haschild(+person,-person).',
    'haschild(-person,+person).',
    'ismarriedto(+person,+person).',
    'ismarriedto(+person,-person).',
    'ismarriedto(-person,+person).',
    'imports(+person,+material).',
    'imports(+person,-material).',
    'imports(-person,+material).',
    'hasmusicalrole(+person,+musicalrole).',
    'hasmusicalrole(+person,-musicalrole).',
    'hasmusicalrole(-person,+musicalrole).',
    'influences(+person,+person).',
    'influences(+person,-person).',
    'influences(-person,+person).',
    'isaffiliatedto(+person,+team).',
    'isaffiliatedto(+person,-team).',
    'isaffiliatedto(-person,+team).',
    'isknownfor(+person,+theory).',
    'isknownfor(+person,-theory).',
    'isknownfor(-person,+theory).',
    'ispoliticianof(+person,+place).',
    'ispoliticianof(+person,-place).',
    'ispoliticianof(-person,+place).',
    'graduatedfrom(+person,+institution).',
    'graduatedfrom(+person,-institution).',
    'graduatedfrom(-person,+institution).',
    'exports(+place,+material).',
    'exports(+place,-material).',
    'exports(-place,+material).',
    'edited(+person,+media).',
    'edited(+person,-media).',
    'edited(-person,+media).',
    'wrotemusicfor(+person,+media).',
    'wrotemusicfor(+person,-media).',
    'wrotemusicfor(-person,+media).'],
    'facebook': ['edge(+person,+person).',
            'edge(+person,-person).',
            'edge(-person,+person).',
            'middlename(+person,+middlename).',
            'middlename(+person,-middlename).',
            'middlename(-person,+middlename).',
            'lastname(+person,+lastname).',
            'lastname(+person,-lastname).',
            'lastname(-person,+lastname).',
            'educationtype(+person,+educationtype).',
            'educationtype(+person,-educationtype).',
            'educationtype(-person,+educationtype).',
            'workprojects(+person,+workprojects).',
            'workprojects(+person,-workprojects).',
            'workprojects(-person,+workprojects).',
            'educationyear(+person,+educationyear).',
            'educationyear(+person,-educationyear).',
            'educationyear(-person,+educationyear).',
            'educationwith(+person,+educationwith).',
            'educationwith(+person,-educationwith).',
            'educationwith(-person,+educationwith).',
            'location(+person,+location).',
            'location(+person,-location).',
            'location(-person,+location).',
            'workwith(+person,+workwith).',
            'workwith(+person,-workwith).',
            'workwith(-person,+workwith).',
            'workenddate(+person,+workenddate).',
            'workenddate(+person,-workenddate).',
            'workenddate(-person,+workenddate).',
            'languages(+person,+languages).',
            'languages(+person,-languages).',
            'languages(-person,+languages).',
            'religion(+person,+religion).',
            'religion(+person,-religion).',
            'religion(-person,+religion).',
            'political(+person,+political).',
            'political(+person,-political).',
            'political(-person,+political).',
            'workemployer(+person,+workemployer).',
            'workemployer(+person,-workemployer).',
            'workemployer(-person,+workemployer).',
            'hometown(+person,+hometown).',
            'hometown(+person,-hometown).',
            'hometown(-person,+hometown).',
            'educationconcentration(+person,+educationconcentration).',
            'educationconcentration(+person,-educationconcentration).',
            'educationconcentration(-person,+educationconcentration).',
            'workfrom(+person,+workfrom).',
            'workfrom(+person,-workfrom).',
            'workfrom(-person,+workfrom).',
            'workstartdate(+person,+workstartdate).',
            'workstartdate(+person,-workstartdate).',
            'workstartdate(-person,+workstartdate).',
            'worklocation(+person,+worklocation).',
            'worklocation(+person,-worklocation).',
            'worklocation(-person,+worklocation).',
            'educationclasses(+person,+educationclasses).',
            'educationclasses(+person,-educationclasses).',
            'educationclasses(-person,+educationclasses).',
            'workposition(+person,+workposition).',
            'workposition(+person,-workposition).',
            'workposition(-person,+workposition).',
            'firstname(+person,+firstname).',
            'firstname(+person,-firstname).',
            'firstname(-person,+firstname).',
            'birthday(+person,+birthday).',
            'birthday(+person,-birthday).',
            'birthday(-person,+birthday).',
            'educationschool(+person,+educationschool).',
            'educationschool(+person,-educationschool).',
            'educationschool(-person,+educationschool).',
            'name(+person,+name).',
            'name(+person,-name).',
            'name(-person,+name).',
            'gender(+person,+gender).',
            'gender(+person,-gender).',
            'gender(-person,+gender).',
            'educationdegree(+person,+educationdegree).',
            'educationdegree(+person,-educationdegree).',
            'educationdegree(-person,+educationdegree).',
            'locale(+person,+locale).',
            'locale(+person,-locale).',
            'locale(-person,+locale).'],
    'yeast2': ['cites(+paper,+paper)',
               'cites(+paper,-paper)',
               'cites(-paper,+paper)',
               'gene(+paper,+gene)',
               'gene(+paper,-gene)',
               'gene(-paper,+gene)',
               'journal(+paper,+journal)',
               'journal(+paper,-journal)',
               'journal(-paper,+journal)',
               'author(+paper,+author))',
               'author(+paper,-author)',
               'author(-paper,+author)',
               'chem(+paper,+chemical)',
               'chem(+paper,-chemical)',
               'chem(-paper,+chemical)',
               'aff(+paper,+institute)',
               'aff(+paper,-institute)',
               'aff(-paper,+institute)',
               'aspect(+paper,+gene,+R)',
               'aspect(-paper,-gene,+R)',
               'aspect(+paper,-gene,-R)',
               'aspect(-paper,+gene,-R)'],
    'fly': ['journal(+paper,+journal)',
            'journal(+paper,-journal)',
            'journal(-paper,+journal)',
            'author(+paper,+author))',
            'author(+paper,-author)',
            'author(-paper,+author)',
            'cites(+paper,+paper)',
            'cites(+paper,-paper)',
            'cites(-paper,+paper)',
            'cites(+paper,+paper)',
            'cites(+paper,-paper)',
            'cites(-paper,+paper)',
            'gene(+paper,+gene)',
            'gene(+paper,-gene)',
            'gene(-paper,+gene)',
            'aspect(+paper,+gene,+R)',
            'aspect(-paper,-gene,+R)',
            'aspect(+paper,-gene,-R)',
            'aspect(-paper,+gene,-R)',
            'gp(+gene,+protein)',
            'gp(+gene,-protein)',
            'gp(-gene,+protein)',
            'genetic(+gene,+gene)',
            'genetic(+gene,-gene)',
            'genetic(-gene,+gene)'
            'physical(+gene,+gene)',
            'physical(+gene,-gene)',
            'physical(-gene,+gene)'],
      }
