
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import psutil
import spacy
import sys
import os

if os.name == 'posix' and sys.version_info[0] < 3:
    import subprocess32 as subprocess
else:
    import subprocess

def call_process(cmd):
    '''Create a subprocess and wait for it to finish. Error out if errors occur.'''
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    pid = p.pid

    (output, err) = p.communicate()  

    #This makes the wait possible
    p_status = p.wait()


def create_fasttext_spacy_file():
    # Download Wikipedia pre-trained FastText model at https://fasttext.cc/docs/en/english-vectors.html
    # When download is over, please create a new directory named fasttext inside TransBoostler's and place the model file in it.

    directory = os.path.abspath('..')
    if(not os.path.exists(os.path.exists('/'.join([directory, 'fasttext/wiki.en.bin'])))):
        
        #Creates new folder to download Word2Vec pre-trained model
        os.mkdir(os.path.join(directory, '/fasttext'))
        
        raise("Please download Wikipedia pre-trained FastText model and place it inside 'fasttext' directory")
        
    else:

        model = KeyedVectors.load_word2vec_format('/'.join([directory, 'fasttext/wiki.en.vec']), binary=False)
        os.mkdir('/'.join([directory, 'fasttext/spacy']))
        model.wv.save_word2vec_format('/'.join([directory,'fasttext/spacy/wiki.txt']), binary=False)
        
        call_process('cd ..; python3 -m spacy init vectors en fasttext/spacy/wiki.txt  fasttext/spacy/ --name en_wiki.vectors --verbose')
    
    return 'SpaCy model created successfully. You can access it by en_googlenews.vectors'
    
create_fasttext_spacy_file()    
    
