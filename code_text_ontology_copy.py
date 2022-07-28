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
        print('done writing to ' + '%s.txt' % item.name + ' file')

# TODO see if the new lines line 33 needs more expansion and how to compensate for 2 columns in pdf if issues

# TODO have as separate functions
# open files in folder ; convert to text ; manipulate the text all separate functions
# with open('%s.txt'%item.name
# TODO why am I creating an output.txt file when above I had files created and named according to the use-case name?
# output.txt looks the same (has the same content and structure) as the ontopaper_usecase.pdf.txt

# TODO change the file name dynamically. According to the name of the pdf that is in the paper_collection_folder
# TODO write the text file from pdftotext in a folder and from there query the folder to open all txt files and work on them

'''
text_article_folder = yourpath / Path(text_article_folder)
text_article_files_in_folder = text_article_folder.iterdir()
for item in text_article_files_in_folder:
    if item.is_file():
        print(item.name)
    with open(item, 'r') as f:
        log = f.readlines()
    text_article = ''
    for line in log:
        text_article += line
    #print(text_article)


# TODO change to the name of the file.

with open('ontopaper_usecase.pdf.txt') as f:
    log = f.readlines()

text_article = ''
for line in log:
    text_article += line
print(text_article)
'''

def workingwithtextfiles(text_article_folder):
    global text_article
    text_article = ''
    text_article_folder = yourpath / Path(text_article_folder)
    text_article_files_in_folder = text_article_folder.iterdir()
    for item in text_article_files_in_folder:
        with open(item) as f:
            print('this is item in text_article_folder', item)
 #           log = f.readlines()
 #           for line in log:
 #               text_article += line
 #           f.close()
        continue
  #  return{'text article folder text': item.name, 'text_article': text_article}
    return {'text article folder text': item.name}

hello = workingwithtextfiles(text_article_folder)
with open('out.txt', 'a+') as f:
    print('Filename:', hello, file=f)  # Python 3.x

# regex for ArrayExpress Accession codes for experiments : E-XXXX-n
# regex = (r"E-[A-Z]{4}-[0-9]*") to find in text_article

# accession_numbers_in_article = str(re.search("E-[A-Z]{4}-[0-9]*", text_article))


#this needs to be fed dynamically from the function using regex to find accession urls in the text article

#TODO Rename this function coz it does more than finding article's accession number
# function that finds the article's accession numbers, does a REST request to return the fetch file (xml file)
#TODO edit this function so that it incluede code from the xml_metadata_processing*=(url) as the code there is more slick
    #TODO and also has writing the response in a text file? is this needed though? to write the xml file in hello.text?
#todo sample code below with exact array express name - remove afterwards
r = requests.get('https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/E-MTAB-1729')
# pprint.pprint(r.content)
arrayexpresscontent = r.content


#TODO feed accession_study_url_to_concatenate from dynamic regex of the article (see code end of the script)

#according to this website this is the url format to use for REST-style queries to retrieve results in XML format
accession_study_url_to_concatenate = "https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/"


# code to get the xml file and parse it and find all the value tags which after will check against the
# PO dev file

rest_request_url = 'https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/E-MTAB-1729'

'''
def QuestionSet1():
    print("Challenge level 1 has being selected.")
    print("Can you translate these words into french?")
    a=input('Q1. Hello! :')
    score = 0
    if 'bonjour' in a.lower(): 
        score = score + 1
        print('Correct!')
    else:
        print('Wrong! '+'Its Bonjour')
        print('You have finished and scored', score, 'out of 10')
    print(score)
QuestionSet1()
'''


def processing_array_express_info(article_text, accession_url):
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

        return {'metadata': metadata, 'metadata score': score}


#TODO REMOVE THIS line which calls the function? as I call it above?
processing_array_express_info(text_article, accession_study_url_to_concatenate)
#this results to a list
#['anthesis', 'CM-82036 resistant parent line', 'NIL1, Fhb1 and Qfhs.ifa-5A resistance alleles', 'NIL2, Fhb1 resistance allele', 'NIL3, Qfhs.ifa-5A resistance allele', 'NIL4, non resistance allele', 'Triticum aestivum', 'spikelet floret', 'Institute for Biotechnology in Plant Production, IFA-Tulln, University of Natural Resources and Life Sciences, A-3430 Tulln, Austria', 'CM-82036 resistant parent line', 'NIL1, Fhb1 and Qfhs.ifa-5A resistance alleles', 'NIL2, Fhb1 resistance allele', 'NIL3, Qfhs.ifa-5A resistance allele', 'NIL4, non resistance allele', 'Fusarium graminearum', 'mock', '30 hour', '50 hour']


#TODO make a function that will find matches between the result of the processing_array_express_info function (i.e.
# the metadata list and the PO_dict dictionary "onto = {}"

# TODO question: is this the best place for this code to be executed and the best way?
# can this be executed in a function whilst it does other stuff done, eg. whilst reading the article line by line?


# pre-processing of text_article
# remove punctuation
def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


parsed_data1 = remove_punctuation(text_article)

#TODO consider if punctuation should stay or give the user option to decide (i.e. depending on what they choose the input of the function to be).

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

tokenised_data = nltk.word_tokenize(text_article)  # parsed_data1 instead of text_article?
#tokenised_data = nltk.word_tokenize(new_log)  # parsed_data1 instead of text_article?

# aim: convert tokenized data which is  a list into a string
# for i in word_tokenize(raw_data):
# print (i)

# Text cleaning: remove stop words.
# stopwords = nltk.corpus.stopwords.words('english')
# ps = nltk.PorterStemmer()

tokens = word_tokenize(parsed_data1)  # list structure, without the punctuation

# remove remaining tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
#TODO remove one variable, change its name as I have two variables as words??? Decide if to keep the stopwords modification


# filter out stop words e.g. of, for .... (but I need some stopwords... e.g. fruit size up to 10% stage
words = [w for w in tokens if not w in stopwords]
print(words)

#filtered_words = [word for word in tokens if word not in stopwords.words('english')]
#print(filtered_words)

# https://machinelearningmastery.com/clean-text-machine-learning-python/
#TODO decide what to do with the stopwords, remove the link and add it as reference in the thesis.

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+')

# examples of keywords in the onto {} (Plant ontologies dictionary) have punctuation, thus keep punctuation in tokens.
# e.g. 'root-derived cultured plant cell': 'PO:0000008'

# create a dictionary with keys the name of the plant ontology and value the PO term.
po_dict = {}
for line in open('plant-ontology-dev.txt'):
    split = line.split("\t")

    # create a dictionary with keys the name of the plant ontology and value the PO term from the plant-onto-dev.txt.
    if len(split) > 2:
        po_dict[split[1]] = split[0]
print('plant ontology dictionary', po_dict)

# TODO should I close the plant-ontology-dev.txt after?

# because many authors describe their research using terms from the "synonyms" and potentially the
# 'definition' column from the plant-ontology-dev then it would be useful to work with a
# dictionary of lists, where the 'key' is the 'id' e.g. PO:00000001 and the 'value' is a list.
# I need a dictionary that maps each key to multiple values.
# PO_dict = {'PO:0000001': ['plant embryo proper', 'an embryonic plant structure .....', 'embri&#243foro (Spanish, ..]'}
# how can I build that from the plant-ontology-dev
# after I construct the PO_dict, how can I use it to find matches of 'name', 'definition', or 'synonyms' in the article?
# and what about finding matches of 'id' in the article? Coz the authors may quote the official PO id in their article


#creates a dictionary of lists with keys the PO id and the values in a list [name, defn, synonyms]
po_dict_of_lists = {}
for line in open('plant-ontology-dev.txt'):
    split = line.split("\t")

    # create a dictionary with keys the name of the plant ontology and value the PO term from the plant-onto-dev.txt.
    if len(split) > 2:
        po_dict_of_lists[split[0]] = split[1:4]
#print('plant ontology dictionary', my_dict)
#creates a dictionary of lists with keys the PO id and the values in a list [name, defn, synonyms]

#TODO then from here (the po_dict_of_lists I need to match in the text with either the 'name'(done above), 'defn', 'synonyms'
# the definition and synonym values are longer and more wordy. I cannot match the exact sentences....it will not
# result into anything. Instead, I can try to find matches of any words in the value entries of 'defn' and 'synonyms'
# look at how I matched with the name and see how I can implement something similar for the po_dict_of_lists
#How to access items in a Python Dictionary of Lists

single_matches = []
for query, onto_id in po_dict.items():
    if query in tokenised_data:
        single_matches.append(query + " | " + onto_id)
print(single_matches)

matches_list = []
for query, onto_id in po_dict.items():
    wordlist = re.sub("[^\w]", " ", query).split()  # separate the individual words in query (of PO dev file) into tokens
    # e.g.query = 'palea development stage' --> 'palea', 'development', 'stage'
    for word in wordlist:
        if word in words:  # words is the tokenised journal article
            matches_list.append(word + " | " + query + " | " + onto_id)
print('this is the matches list', matches_list)

# this results in fussy matching. Brings all the keys in the  PO dev file with names that contain one of those words
# e.g. plant, results in many matches (plant embryo proper, in vitro plant structure, cultured plant cell).
# found the word in manuscript: whole | query: whole plant fruit formation stage 70% to final size | onto_id: PO:0007027
# maybe also include the complete sentence this matching came from in order to
# allow the reader to assess if it is to their interest that sentence or not.

# match the bigrams and the keys in the onto dictionary
bigram = list(ngrams(tokenised_data, 2))
res = [' '.join(tups) for tups in bigram] #['. BMC', 'BMC Genomics', 'Genomics 2013', 'RESEARCH ARTICLE', 'ARTICLE Open']

#TODO FIX THIS :: found query match in new res bigram match  | Systems Biology | stem

for query, onto_id in po_dict.items():
    if query in res:
        print('found query match with bigram match', " | " + onto_id + " | " + query)

# TODO have the code return all PO ids, that contain a word found in the tokenised data
#   for example, if in the tokenised_data of the article we have the word palea, return apart from the exact match
#   return e.g. 'palea apiculus' as well. Meaning find matches with the onto dictionary with any of the words in the
#   query key.


'''
#test code for applying ngrams

sentence = 'this is a foo bar sentences and i want to ngramize it'

n = 6
sixgrams = ngrams(sentence.split(), n)

for grams in sixgrams:
    print(grams)
'''

n = 3

#TODO decide about the punctuation

threegrams = list(ngrams(tokenised_data, 3)) #I need the punctuation as name can be "fruit size up to 10% stage"
print('this is threegrams', threegrams)
# so in order to find matches of onto dictionary and the 3grams, the 3grams must not be separated by comma. So need to join.

combine3gram = [' '.join(tups) for tups in threegrams]
print("The joined data combine3grams is : " + str(combine3gram))


#TODO eventually combine all the n-grams, in a loop where it starts from 1 till 6 grams

threegram_matches = []
for query, onto_id in po_dict.items():
    if query in combine3gram:
        threegram_matches.append(query + " |" + onto_id)
print(threegram_matches)
#  else:
#      print('didnt find a matching ontology term')


#to do the match with the synonym and definition columns
# raw data is the PO ontology file
raw_data = open('plant-ontology-dev.txt').read()

parsed_data2 = raw_data.replace('\n', '\t').split('\t')
definition_list = parsed_data2[2::6]

synonym_list = parsed_data2[3::6]
print("ontology synonym list is", synonym_list[0:20])

def replace_certain_punctuations(list):
    records = [rec.replace('-', ' ').replace('_', ' ').replace('.', ' ') for rec in list]
    return records

replace_certain_punctuations(synonym_list)



#######################################################################################################################

#TODO make a function that will find matches between the result of function processing_array_express_info function (i.e.
# the metadata list and the PO_dict dictionary "onto = {}"

print("Running processing array express metadata next")

xml_metadata = processing_array_express_info(text_article, accession_study_url_to_concatenate)
print('xml_metadata list', xml_metadata)

print('metadata score is', xml_metadata['metadata score']) #this will be added with the other scores of each function

for query, onto_id in po_dict.items():
    if query in xml_metadata:
        print('found single word matches between PO onto dict and xml metadata:', query, onto_id)
#from our use-case example this returns one match:
#found single word matches between PO onto dict and xml metadata spikelet floret PO:0009082

#TODO examine the ontologies (i.e. ontological metadata) matched to PO dict in the paper with the ones found in
# the xml_metadata, e.g. in the use-case example - spikelet floret PO:OO9082. Does the article contain the words
# spikelet floret? Ways to go about this: 1. identify it is a 2 word string (look through the bigram list derived from
# the text_article processing. Or make the article into bigrams and then check if the bigram spikelet floret is there.
# problem: how can this be scalable? It is not always that I will have bigrams? So write a line that checks the length
# of the string resulted? Then compare that with the relevant n-gram?
# "res" variable is the bigram of text_article. So in this example, I can compare "res" and 'spikelet floret'.
# or if both "res" and the list produced from the above code (e.g. list_made = [spiklet floret] and find common strings
# as well as useful to find different elements between the 2 lists, such that the pipeline can say: the xml_metadata
# contained the ontological term "spikelet floret" but the article did not have it. The xml_metadata contained "anthesis"
# but it didn't contain pilea.
# because the ontological terms in the xml_metadata could also be the non-standard PO ontological names (one example with
# our use-case is "anthesis". 'Anthesis" is the synonym of "flowering stage". So the pipeline can also report that
# ontological terms in the xml_metadata have been used, but they used the synonym.

#TODO so then possibly I can compare the xml_metadata = [] with the PO_dict that contains the synonym, or definition
# columns of the Plant ontology dev text file.

#easier...compare the bigram matches from the article with the xml_metadata.
#easier...all the n-gram matches from the text article to the PO dict, append them to a list.
# then compare that list with the xml_metadata list. Because if a match was done in the article and PO dict comparison
#stage, then this list of matches can be compared with the xml_metadata list matches.


#TODO create a list and append all the text article n-gram matches with the PO dict. (have a function that searches
# for n-gram matches until n=6.
# compare that appended list with the xml_metadata list.



#########################################################################################
# Data accessibility check stage of the code
#########################################################################################

# read text, find and print the phrases which include the data_reproducibility_keywords.
# TODO if any such data_reproducibility_keywords have been found, then add a point to the reproducibility metrics score

# todo how to calculate the reproducibility metrics score.
# add one point each time in the manuscript is found: data_reproducibility_keywords,

# from https://simply-python.com/2014/03/14/saving-output-of-nltk-text-concordance/

def get_all_phrases_containing_tar_wrd(target_word, tar_passage, left_margin=10, right_margin=10):
    """
        Function to get all the phases that contain the target word in a text/passage tar_passage.
        Workaround to save the output given by nltk Concordance function

        str target_word, str tar_passage int left_margin int right_margin --> list of str
        left_margin and right_margin allocate the number of words/punctuation before and after target word
        Left margin will take note of the beginning of the text
    """
    ## Create list of tokens using nltk function
    tokens = nltk.word_tokenize(tar_passage)

    ## Create the text of tokens
    text = nltk.Text(tokens)

    ## Collect all the index or offset position of the target word
    c = nltk.ConcordanceIndex(text.tokens, key=lambda s: s.lower())

    ## Collect the range of the words that is within the target word by using text.tokens[start;end].
    ## The map function is use so that when the offset position - the target range < 0, it will be default to zero
    concordance_txt = (
    [text.tokens[list(map(lambda x: x - 5 if (x - left_margin) > 0 else 0, [offset]))[0]:offset + right_margin]
     for offset in c.offsets(target_word)])

    ## join the sentences for each of the target phrase and return it
    return [''.join([x + ' ' for x in con_sub]) for con_sub in concordance_txt]


data_reproducibility_keywords = ['accession', 'data', 'Supporting', 'available', 'repository', 'GO', 'EBI',
                                 'ArrayExpress', 'PO', 'sequences', 'expression', 'snps', 'genes', 'wheat', 'rice']
phrases_from_article = []
for word in data_reproducibility_keywords:
    phrases_from_article = get_all_phrases_containing_tar_wrd(word, text_article)
    print('phrases from text article:', word,
          phrases_from_article)  # it doesn't return all the occurences or the complete
    # sentences because it is two column text.


# returning sentences containing particular phrases: e.g. "Supporting data"
def regex_search(filename, term):
    searcher = re.compile(term + r'([^\w-]|$)').search
    with open(filename, 'r') as source, open("new.txt", 'w') as destination:
        for line in source:
            if searcher(line):
                destination.write(line)  # fclose?


find_phrases = regex_search('ontopaper_usecase.pdf.txt', 'accession number')

# return complete sentences if they contain 'wanted' word.
txt = "I like to eat apple. Me too. Let's go buy some apples."
hello = [sentence + '.' for sentence in txt.split('.') if 'apple' in sentence]
print(hello)
# ['I like to eat apple.', " Let's go buy some apples."]

'''
#we got two column text, we need to make into one column so that the correct lines are captured.
def remove_newlines(fname):
    flist = open(fname).readlines()
    with open("new11.txt", 'w') as f:
        do_this = [s.rstrip('\n') for s in flist]
        f.write(str(do_this)) #
        f.close()
    #return [s.rstrip('\n') for s in flist]


remove_newlines1 = remove_newlines('ontopaper_usecase.pdf.txt')
#['Kugler et al. BMC Genomics 2013, 14:728', 'http://www.biomedcentral.com/1471-2164/14/728', '  RESEARCH ARTICLE                                                                                                                                     Open Access', 'Quantitative trait loci-dependent analysis of a', 'gene co-expression network associated with',
# this is not what I want. It creates a list of strings where \n was there before.

hello1 = [sentence + '.' for sentence in remove_newlines1.split('.') if 'data' in sentence]
print('this is hello1', hello1)
'''

data = ''
with open('ontopaper_usecase.pdf.txt', 'r') as file:
    data = file.read().replace('\n', '').replace('\t', '')  # parsed_data2 = raw_data.replace('\n','\t').split('\t')
    with open('new22.txt', 'w') as file1:
        file1.write(data)
        file1.close()
    file.close()

hello22 = [sentence + '.' for sentence in data.split('.') if 'rice' in sentence]
print(hello22)

'''
with open('demofile.txt','w') as f:
   f.write(remove_newlines(('ontopaper_usecase.pdf.txt')))
   f.close()
'''

# print(remove_newlines("ontopaper_usecase.pdf.txt"))

'''
def extract_phases(tokens, wordlist):
    all_phrases = []
    text = nltk.Text(tokens)
    for word in wordlist:
        phrases = get_all_phrases_containing_tar_wrd(word, text)
        if phrases:
            all_phrases.append(phrases)
    print('all word list')
    return all_phrases
'''

# regex for ArrayExpress Accession codes for experiments : E-XXXX-n
# regex = (r"E-[A-Z]{4}-[0-9]*") to find in text_article

accession_numbers_in_article = re.search("E-[A-Z]{4}-[0-9]*", text_article)
print(accession_numbers_in_article)


#TODO device scoring system for reproducibility metrics (reproducibility scoring)

'''

#how to have a function return an object with many values

class ReturnValue(obj):
   def __init__(self, y0, y1, y2):
      self.y0 = y0
      self.y1 = y1
      self.y2 = y2

def g(x):
   y0 = x + 1
   y1 = x * 3
   y2 = y0 ** 3
   return ReturnValue (y0, y1, y2)

'''

