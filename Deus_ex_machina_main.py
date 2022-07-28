__author__ = 'Evanthia_Kaimaklioti Samota'

# !/usr/bin/env python
# # # coding: utf-8


import pandas as pd
import os, os.path, pdftotext, re, string, nltk, argparse, sys, functools, operator
import glob, os
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pathlib import Path
import urllib.request, urllib.error, urllib.parse
import requests, pprint
import bs4
from bs4 import BeautifulSoup
from itertools import repeat



class File(object):

    def __init__(self, name):
        self.name = name

    def convert_pdf_to_text(self, name):
        with open(name, 'rb') as f:
            pdf = pdftotext.PDF(f, 'secret')
        for page in pdf:
            f = open('%s.txt' % self.name, "w+")
            f.write(" ".join(pdf))  # I dont want new lines so f.write("\n\n".join(pdf))
            f.close()
        return ('%s.txt' % self.name)

    def create_text_file(self, name):
        with open(name) as f:
            log = f.readlines()
        text_article = ''
        for line in log:
            text_article += line
        tokenised_data = nltk.word_tokenize(text_article)
        return tokenised_data

    # pre-processing of text_article
    # remove punctuation
    def remove_punctuation(self, txt):
        txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
        return txt_nopunct

    def find_unigrams(self, data_tokenised, dict_po):
        single_matches = []
        for query, onto_id in dict_po.items():
            if query in data_tokenised:
                single_matches.append(query + " | " + onto_id)
        print(single_matches)
        return single_matches

    # match the bigrams and the keys in the onto dictionary
    def find_bigrams(self, data_tokenised, dict_po):
        bigram = list(ngrams(data_tokenised, 2))
        res = [' '.join(tups) for tups in bigram]  # ['. BMC', 'BMC Genomics', 'Genomics 2013', 'RESEARCH ARTICLE', 'ARTICLE Open']
        bigram_matches = []
        for query, onto_id in dict_po.items():
            if query in res:
                print('found query match with bigram match', " | " + onto_id + " | " + query)
                bigram_matches.append(query + " | " + onto_id)
        print(bigram_matches)
        return(bigram_matches)

    def find_ngrams(self, data_tokenised, dict_po):
        ngram_matches = []
        n=6
        for i in range(1,n+1):
            n_grams = list(ngrams(data_tokenised, i))
            print(i, n_grams)
            res = [' '.join(tups) for tups in n_grams]
            print('res', i, res)
            for query, onto_id in dict_po.items():
                if query in res:
                    print('found ontology match with n_gram match', " | ", i, onto_id + " | " + query)
                    ngram_matches.append(query + " | " + onto_id)
        print(ngram_matches)
        return ngram_matches

    # line 422 from other script: def get_all_phrases_containing_tar_wrd(target_word, tar_passage, left_margin=10, right_margin=10):


# create a dictionary with keys the name of the plant ontology and value the ID.
po_dict = {}
for line in open('plant-ontology-dev.txt'):
    split = line.split("\t")
    if len(split) > 2:
        po_dict[split[1]] = split[0]

file1 = File("ontology_usecase2.pdf")
print(file1.name)

convertpdf = file1.convert_pdf_to_text(file1.name)
textarticlecreation = file1.create_text_file(convertpdf)

# the_single_matches = file1.find_unigrams(textarticlecreation, po_dict)
# the_bigram_matches = file1.find_bigrams(textarticlecreation, po_dict)
ngram_matches = file1.find_ngrams(textarticlecreation, po_dict)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+')

# TODO add the punctuation and stopwords code
# examples of keywords in the onto {} (Plant ontologies dictionary) have punctuation, thus keep punctuation in tokens.
# e.g. 'root-derived cultured plant cell': 'PO:0000008'