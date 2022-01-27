#!/bin/bash
echo "Installing project dependencies.."

pip3 install ekphrasis
pip3 install scipy==1.6.3
pip3 install gensim==4.0.1
pip3 install git+git://github.com/ThaisLuca/boostsrl-python-package.git
pip3 install git+git://github.com/ThaisLuca/wmd-relax
pip3 install -U scikit-learn==0.24.2
pip3 install psutil
pip3 install unidecode
pip3 install pandas
pip3 install pyemd


pip3 install -U pip setuptools wheel
pip3 install -U spacy

echo "Installation is complete!"

echo "Generating SpaCy model from FastText and Word2Vec"

python3 build_spacy_fasttext_model.py
#python3 build_spacy_word2vec_model.py

echo "All set!"
