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

    def create_text_article(self, name):
        with open(name) as f:
            log = f.readlines()
        text_article = ''
        for line in log:
            text_article += line
        return text_article
    # TODO the create_text_article should return just the text_article as we need it for other functions
    #   maybe see if create_text_file function can run the create_text_article function first and add the extra line for tokenisation

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


    def get_all_phrases_containing_tar_wrd(self, target_word, tar_passage, left_margin=30, right_margin=30):
        ## Create list of tokens using nltk function
        tokens = nltk.word_tokenize(tar_passage)

        ## Create the text of tokens
        text = nltk.Text(tokens)

        ## Collect all the index or offset position of the target word
        c = nltk.ConcordanceIndex(text.tokens, key=lambda s: s.lower())

        ## Collect the range of the words that is within the target word by using text.tokens[start;end].
        ## The map function is use so that when the offset position - the target range < 0, it will be default to zero
        concordance_txt = ([text.tokens[list(map(lambda x: x - 5 if (x - left_margin) > 0 else 0, [offset]))[0]:offset + right_margin] for offset in c.offsets(target_word)])

        ## join the sentences for each of the target phrase and return it
        return [''.join([x + ' ' for x in con_sub]) for con_sub in concordance_txt]


    def processing_array_express_info(self, article_text, accession_url):
        accession_numbers_in_article = re.findall("E-[A-Z]{4}-[0-9]*", article_text)
        score = 0
        s = set()
        set_article_accession_numbers = set(accession_numbers_in_article)  # {'E-MTAB-1729'}
        #print(set_article_accession_numbers)  # TODO if the script cannot find any accession numbers & put negative score?
        if len(set_article_accession_numbers) == 0:
            #print('could not find any ArrayExpress accession numbers here is the score', score)
            return None, score
        else:
            score = score + 1
            print('I could find ArrayExpress accession numbers, here is the score', score)
            for accession_number in set_article_accession_numbers:
                api_url_concatenated = accession_url + str(accession_number)
                getxml = requests.request('GET', api_url_concatenated)
                file = open('%s.txt' % accession_number, 'w')
                file.writelines(getxml.text)
                file.close()
                soup = bs4.BeautifulSoup(getxml.text, 'xml')
                # print(soup.prettify())
                metadata = []
                for hit in soup.find_all("value"):
                    metadata.append(hit.text.strip())
                print('this is metadata', metadata)
                return metadata


# TODO could be better to make another function that assesses the output of function processing_array_express_info and depending on that compute the score
'''
def processing_array_express_info1(article_text, accession_url):
    accession_numbers_in_article = re.findall("E-[A-Z]{4}-[0-9]*", article_text)
    score = 0
    s = set()
    set_article_accession_numbers = set(accession_numbers_in_article)  #{'E-MTAB-1729'}

    if set_article_accession_numbers == (s == set()):
        print(score)
    else:
        score = score + 1

    for accession_number in set_article_accession_numbers:
        api_url_concatenated = accession_url + str(accession_number)
        getxml = requests.request('GET', api_url_concatenated)
        file = open('response.txt', 'w')
        file.writelines(getxml.text)
        file.close()

        soup = bs4.BeautifulSoup(getxml.text, 'xml')

        metadata = []
        for hit in soup.find_all("value"):
            metadata.append(hit.text.strip())

        return(metadata)
        #return {'metadata': metadata, 'metadata score': score}
'''


# returning sentences containing particular phrases: e.g. "Supporting data"
def regex_search(filename, term):
    searcher = re.compile(term + r'([^\w-]|$)').search
    with open(filename, 'r') as source, open("new.txt", 'w') as destination:
        for line in source:
            if searcher(line):
                destination.write(line)  # fclose?

# create a dictionary with keys the name of the plant ontology and value the ID.
po_dict = {}
for line in open('plant-ontology-dev.txt'):
    split = line.split("\t")
    if len(split) > 2:
        po_dict[split[1]] = split[0]

#ontopaper_usecase.pdf
#ontology_usecase2.pdf
file1 = File("ontology_usecase2.pdf")
print(file1.name)

convertpdf = file1.convert_pdf_to_text(file1.name)
textarticlecreation = file1.create_text_file(convertpdf)
text_article = file1.create_text_article(convertpdf)

# the_single_matches = file1.find_unigrams(textarticlecreation, po_dict)
# the_bigram_matches = file1.find_bigrams(textarticlecreation, po_dict)
ngram_matches = file1.find_ngrams(textarticlecreation, po_dict)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+')

data_reproducibility_keywords = ['accession', 'accessions', 'data', 'Supporting', 'available', 'repository', 'GO', 'EBI',
                                     'ArrayExpress', 'PO', 'sequences', 'expression', 'snps', 'genes', 'wheat', 'rice']
phrases_from_article = []
for word in data_reproducibility_keywords:
    phrases_from_article = file1.get_all_phrases_containing_tar_wrd(word, text_article)
    print('phrases from text article:', word, phrases_from_article) # it doesn't return all the occurences or the complete
    # sentences because it is two column text.

# TODO run the working version of code_text_ontology_copy code to see what the function fetching the phraes does, coz here it returns empty list
# TODO 1. copy from Github the old code_text_ontology_copy.py code, 2. run it, 3. compare the outputs here 4. solve the code you go it

# TODO how to compensate for the two column text? is there a parameter with pdf2text that can bypass this?


#find_phrases = file1.regex_search('ontopaper_usecase.pdf.txt', 'accession number')


# TODO add the punctuation and stopwords code
# examples of keywords in the onto {} (Plant ontologies dictionary) have punctuation, thus keep punctuation in tokens.
# e.g. 'root-derived cultured plant cell': 'PO:0000008'

# for studies such as ontopaper2 which dont have Array express accession numbers, but maybe including SRP numbers
#https://www.ebi.ac.uk/ena/browser/view/PRJDB2496?show=reads
#https://www.ebi.ac.uk/ena/browser/api/xml/DRP000768?download=true

accession_study_url_to_concatenate = "https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/"
xml_metadata = file1.processing_array_express_info(text_article, accession_study_url_to_concatenate)

'''
for query, onto_id in po_dict.items():
    if query in xml_metadata:
        print('found single word matches between PO onto dict and xml metadata:', query, onto_id)
'''
# if xml_metadata is empty it throws an error, so need to catch this error so the code doesn't break/stop


'''
for query, onto_id in po_dict.items():
    a = []
    if len(xml_metadata1) == len(a):
        print('xml metadata list is empty')
    else:
        if query in xml_metadata1:
            print('found single word matches between PO onto dict and xml metadata:', query, onto_id)
'''
'''
if xml_metadata[0] is None:
    print('xml metadata is empty and score for this function is:', xml_metadata[1])
else:
    score_for_xml_ontology_matching = 0
    for query, onto_id in po_dict.items():
        if query in xml_metadata:
            print('found single word matches between PO onto dict and xml metadata:', query, onto_id, score_for_xml_ontology_matching + 1)
        else:
            print('no matches were found in the xml metadata file in ArrayExpress and the po_dict and the score for this is', score_for_xml_ontology_matching)
'''
#TODO possibly it's better to have a separate function that takes the functions and computes the scores according to the output of each function.
# and then puts the results in SQLite or CSV for the user to view and analyse

score_for_xml_ontology_matching = 0
if xml_metadata[0] is not None:
    #print(xml_metadata[0])
    #print(len(xml_metadata))
    if len(xml_metadata) == 0:
        print('xml metadata is empty')
    else:
        for query, onto_id in po_dict.items():
            if query in xml_metadata:
                print('found single word matches between PO onto dict and xml metadata:', query, onto_id)
                print('the score for this function is:', score_for_xml_ontology_matching + 1)
else:
    # this runs
    print('xml_metadata variable stores a None value. This function cannot run. Score for this is:', score_for_xml_ontology_matching)




'''
if xml_metadata is not None:
    print(xml_metadata[0])
    #print(len(xml_metadata))
    if len(xml_metadata) == 0:
        print('xml metadata is empty')
    else:
        for query, onto_id in po_dict.items():
            if query in xml_metadata:
                print('found single word matches between PO onto dict and xml metadata:', query, onto_id)
else:
    # this runs
    print('xml_metadata variable stores a None value. This function cannot run. Score for this is 0. ')
'''
'''
if len(xml_metadata) == 0:
    print('xml metadata is empty')
else:
    for query, onto_id in po_dict.items():
        if query in xml_metadata:
            print('found single word matches between PO onto dict and xml metadata:', query, onto_id)
'''

#TODO apply for ngrams again? review code. and what I want to do
