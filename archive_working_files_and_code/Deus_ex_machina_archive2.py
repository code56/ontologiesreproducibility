__author__ = 'Evanthia_Kaimaklioti Samota'

# !/usr/bin/env python
# # # coding: utf-8


import pandas as pd
import os, os.path, pdftotext, re, string, nltk, argparse, sys, functools, operator
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pathlib import Path
import urllib.request, urllib.error, urllib.parse
import requests, pprint
import bs4
from bs4 import BeautifulSoup

# Pre-requisite: put the pdf version of the articles you want to process in the papers_collection_folder

yourpath = os.getcwd()
data_folder = Path("papers_collection_folder")
text_article_folder = Path("text_article_folder")

folder = yourpath / Path(data_folder)
files_in_folder = data_folder.iterdir()
for item in files_in_folder:
    if item.is_file():
        print(item.name)
    with open(item, 'rb') as f:
        pdf = pdftotext.PDF(f, 'secret')
        for page in pdf:
            completeName = os.path.join(yourpath / Path(text_article_folder), '%s.txt' % item.name)
            f = open(completeName, "w+")
            f.write(" ".join(pdf))  # I dont want new lines so f.write("\n\n".join(pdf))
            f.close()
        print('hello ' + '%s.txt' % item.name + ' file')

import code_text_ontology_copy.py


print("hello")

#Create a function that encapsulates what you want to do to each file.

import os.path

def parse_pdf(filename):
    "Parse a pdf into text"
    content = getPDFContent(filename)
    encoded = content.encode("utf-8")
    ## split of the pdf extension to add .txt instead.
    (root, _) = os.path.splitext(filename)
    text_file = open(root + ".txt", "w")
    text_file.write(encoded)
    text_file.close()
#Then apply this function to a list of filenames, like so:

for f in files:
    parse_pdf(f)