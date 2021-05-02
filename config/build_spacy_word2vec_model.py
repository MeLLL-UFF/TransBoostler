
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


def create_word2vec_spacy_file():
    # Download Google News pre-trained Word2Vec model at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    # When download is over, please create a new directory named word2vec inside TransBoostler's and place the model file in it.

    directory = os.path.abspath('..')
    if(not os.path.exists(os.path.exists('/'.join([directory, '/word2vec/GoogleNews-vectors-negative300.bin'])))):
        
        #Creates new folder to download Word2Vec pre-trained model
        os.mkdir('/'.join([directory, '/word2vec']))
        
        raise("Please download Google News pre-trained Word2Vec model and place it inside 'word2vec' directory")
        
    else:
        
        model = KeyedVectors.load_word2vec_format('/'.join([directory, '/word2vec/GoogleNews-vectors-negative300.bin']), binary=True, unicode_errors='ignore')
        
        if(not os.path.exists(os.path.exists('/'.join([directory, 'word2vec/spacy'])))):
            os.mkdir('/'.join([directory, 'word2vec/spacy']))

        model.save_word2vec_format('/'.join([directory, '/word2vec/spacy/googlenews.txt']), binary=False)
        
        call_process('cd ..; python3 -m spacy init vectors en word2vec/spacy/googlenews.txt  word2vec/spacy/ --name en_googlenews.vectors --verbose')
    
    return 'SpaCy model created successfully. You can access it by en_googlenews.vectors'
    
    
    
create_word2vec_spacy_file()
