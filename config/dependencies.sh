#!/bin/bash
echo "Installing project dependencies.."

pip3 install ekphrasis
pip3 install gensim
pip3 install git+git://github.com/ThaisLuca/boostsrl-python-package.git
pip3 install psutil
pip3 install unidecode
pip3 install pandas
pip3 install pyemd
pip3 install wmd

pip3 install -U pip setuptools wheel
pip3 install -U spacy

echo "Installation is complete!"

echo "Generating SpaCy model from FastText and Word2Vec"

python3 build_spacy_fasttext_model.py
python3 build_spacy_word2vec_model.py

echo "All set!"
