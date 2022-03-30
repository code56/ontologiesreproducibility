__author__ = 'Evanthia_Kaimaklioti Samota'

# !/usr/bin/env python
# coding: utf-8

# In[7]:

import pandas as pd
import os, os.path, pdftotext, re, string, nltk, argparse, sys, functools, operator
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pathlib import Path
import urllib.request, urllib.error, urllib.parse
import requests, pprint

r = requests.get('https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/E-MTAB-1729')
# pprint.pprint(r.content)
arrayexpresscontent = r.content


# work with an html version of the paper

def fromurltotext(url):
    response = urllib.request.urlopen(url)
    webContent = response.read()

    # print(webContent)
    # save the HTML file locally

    f = open('paper.html', 'wb')
    f.write(webContent)

    # Python - How to read HTML line by line into a list
    with open("paper.html") as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    # want to get the text from the HTML
    with open('paper.html') as f:
        log = f.readlines()

    new_log = ''
    for line in log:
        new_log += line

    f = open('hello.txt', "w+")  # must make it so the name changes each time
    f.write(" ".join(log))  # I dont want new lines so f.write("\n\n".join(pdf))
    f.close()
    print('done writing to hello.txt')
    # print("this is new log")
    # print(new_log) #-this works

    # put this in a function that will read many articles, not just one.
    # add the code of converting the pdf to text


url = 'https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-728'
fromurltotext(url)

yourpath = os.getcwd()

data_folder = Path("papers_collection_folder")  # can be changed to user input data
folder = yourpath / Path(data_folder)
print("this is directory_to_folder", folder)
files_in_folder = data_folder.iterdir()
for item in files_in_folder:
    if item.is_file():
        print(item.name)
    with open(item, 'rb') as f:
        pdf = pdftotext.PDF(f, 'secret')
        for page in pdf:
            # print(page)
            f = open('%s.txt' % item.name, "w+")  # must make it so the name changes each time
            f.write(" ".join(pdf))  # I dont want new lines so f.write("\n\n".join(pdf))
            f.close()
        print('done writing to ' + '%s.txt' % item.name + ' file')

# have as separate functions
# start with the functions ('what do I want the code to do').
# open files in folder ; convert to text ; manipulate the text all separate functions
# with open('%s.txt'%item.name

with open('output.txt') as f:
    log = f.readlines()

text_article = ''
for line in log:
    text_article += line

# regex for ArrayExpress Accession codes for experiments : E-XXXX-n
# regex = (r"E-[A-Z]{4}-[0-9]*") to find in text_article

# accession_numbers_in_article = str(re.search("E-[A-Z]{4}-[0-9]*", text_article))
# works and returns: <re.Match object; span=(71082, 71093), match='E-MTAB-1729'>
# so take the match object and do an EBI API search using this accession number


accession_numbers_in_article_list = re.findall("E-[A-Z]{4}-[0-9]*", text_article)
print(accession_numbers_in_article_list)  # returns a list of accession numbers ['E-MTAB-1729', 'E-MTAB-1729']

accession_study_url_to_concatenate = "https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/"

# in this case where accession_study_url_to_concatenate is ['E-MTAB-1729', 'E-MTAB-1729']
# should I put a check to see if the list should remove duplicate accession numbers?
# so that the for loop below is not running twice for the same accession number?


for accession_number in accession_numbers_in_article_list:
    api_url_concatenated = accession_study_url_to_concatenate + str(accession_number)
    print(api_url_concatenated)
    response = requests.get(api_url_concatenated)
    print(response.text)  # put this in a varialbe


# question: is this the best place for this code to be executed and the best way?
# can this be executed in a function whilst it does other stuff done whilst reading the article line by line?

# requests.get('https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/E-MTAB-1729') returns Response 200 - good

# api_url_concatenated = accession_study_url_to_concatenate + accession_numbers_in_article
# print(api_url_concatenated)
# response=requests.get(api_url_concatenated)
# print(response.text)
# this will print the xml format of the accession number file
# or same as print(response.content)


# then what do I do with the xml file?
# what am I looking to find from it?
# compare the metadata from the document with the one recorded on the xml response file?


# pre-processing of text_article
# remove punctuation

def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


parsed_data1 = remove_punctuation(text_article)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

tokenised_data = nltk.word_tokenize(text_article)  # parsed_data1 instead of text_article?
print('tokenised data', tokenised_data)

# aim convert tokenized data which is  a list into a string
# for i in word_tokenize(raw_data):
# print (i)

# Text cleaning: remove stop words.
# stopwords = nltk.corpus.stopwords.words('english')
# ps = nltk.PorterStemmer()

tokens = word_tokenize(parsed_data1)
print('this is tokens', tokens[:100])  # list structure

# remove remaining tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]

# filter out stop words
words = [w for w in tokens if not w in stopwords]
print('this is words after removing stop words', words[:100])

# https://machinelearningmastery.com/clean-text-machine-learning-python/


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+')

# compare a list_of_bigrams with the plant ontology dictionary. However, the keys in the onto{} are not bigrams -
# i.e not two tokens. Then how do I do the comparison by ignoring the comma? Would I have to make a new onto dictionary
# with tokens separated by coma for the keys? or can I make a new list of bigrams NOT separated by commas?

# working on the plant-ontology-dev.txt
# examples of keywords in the onto {} (Plant ontologies dictionary) which have punctuation
# 'root-derived cultured plant cell': 'PO:0000008'
# create a dictionary with keys the name of the plant ontology and value the PO term.

onto = {}
for line in open('plant-ontology-dev.txt'):
    split = line.split("\t")

    # create a dictionary with keys the name of the plant ontology and value the PO term from the plant-onto-dev.txt.
    if len(split) > 2:
        onto[split[1]] = split[0]
print('plant ontology dictionary', onto)
# should I close the plant-ontology-dev.txt after?

# because many authors describe their research using terms from the "synonyms" and potentially the
# 'definition' column from the plant-ontology-dev then it would be useful to work with a
# dictionary of lists, where the 'key' is the 'id' e.g. PO:00000001 and the 'value' is a list.
# I need a dictionary that maps each key to multiple values.
# PO_dict = {'PO:0000001': ['plant embryo proper', 'an embryonic plant structure .....', 'embri&#243foro (Spanish, ..]'}
# how can I build that from the plant-ontology-dev
# after I construct the PO_dict, how can I use it to find matches of 'name', 'definition', or 'synonyms' in the article?
# and what about finding matches of 'id' in the article? Coz the authors may quote the official PO id in their article


po_dict_of_lists = {}
for line in open('plant-ontology-dev.txt'):
    split = line.split("\t")

    # create a dictionary with keys the name of the plant ontology and value the PO term from the plant-onto-dev.txt.
    if len(split) > 2:
        po_dict_of_lists[split[0]] = split[1:4]
#print('plant ontology dictionary', my_dict)
#creates a dictionary of lists with keys the PO id and the values in a list [name, defn, synonyms]

#then from here (the po_dict_of_lists I need to matches in the text with either the 'name', 'defn', 'synonyms'
# the 'name' is easy to do, and is done
# the definition and synonym values are longer and more wordy. I cannot match the exact sentences....it will not
# result into anything. Instead, I can try to find matches of any words in the value entries of 'defn' and 'synonyms'
# look at how I matched with the name and see how I can implement something similar for the po_dict_of_lists




# would be good to be searching for terms in whole text except in the references.
for query, onto_id in onto.items():
    if query in tokenised_data:
        print('found single word matches', query, onto_id)
# this only returns one word matches for keys, e.g. found query lemma PO:0009037


matches_list = []
for query, onto_id in onto.items():
    wordlist = re.sub("[^\w]", " ", query).split()  # separate the individual words in query into tokens
    # e.g.query = 'palea development stage' --> 'palea', 'development', 'stage'
    for word in wordlist:
        if word in words:  # words is the tokenised journal article
            matches_list.append(word + " | " + query + " | " + onto_id)
# print('this is the matches list', matches_list)

# this results in fussy matching
# found the word in manuscript: whole | query: whole plant fruit formation stage 70% to final size | onto_id: PO:0007027
# maybe also include the complete sentence this matching came from in order to
# allow the reader to assess if it is to their interest that sentence or not.

# match the bigrams and the keys in the onto dictionary


# compare a list_of_bigrams with the plant ontology dictionary. However, the keys in the onto{} are not bigrams -
# i.e not two tokens. Then how do I do the comparison by ignoring the comma? Would I have to make a new onto dictionary
# with tokens separated by coma for the keys? or can I make a new list of bigrams NOT separated by commas?

# joining tuple elements (i.e. removing the separating commas)
# using join() + list comprehension

bigram = list(ngrams(tokenised_data, 2))
# print('this is bigram before removing punctuation from article text', bigram)

res = [' '.join(tups) for tups in bigram]
print("The joined data res is : " + str(res))
# result --> The joined data res is : ['. BMC', 'BMC Genomics', 'Genomics 2013', 'RESEARCH ARTICLE', 'ARTICLE Open']
# find matches between the onto_dict and the res
print(type(res))

new_res_testing = ['Kugler et', 'et al', 'al .', '. BMC', 'BMC Genomics', 'whole plant', 'plant is']
# new_res = []
# new_res = res.append('whole plant')

for query, onto_id in onto.items():
    for words in new_res_testing:
        if query in words:
            print('found query match in res bigrams', " | " + words + " | " + query)

# lateral root, vs lateral root tip, vs all occurances of keys with the word lateral in it

string = 'plant embryo proper'
arr = [x.strip() for x in string.strip('[]').split(' ')]
# result -> ['plant', 'embryo', 'proper']


mystr = 'This is a string, with words!'
wordList = re.sub("[^\w]", " ", mystr).split()
print('this is wordList', wordList)

# This is the wall of the microsporangium.
# vs I found in my garden wall a microsporangium (not wanted).

list_of_bigrams_testing = [('microsporangium', 'wall'), ('whole', 'plant')]
for query, onto_id in onto.items():
    res = [' '.join(tups) for tups in list_of_bigrams_testing]
    if query in res:
        print('found query in the res', query, onto_id)

# print('this is res', res)
# if query in a:
#    print(a, onto_id)
# elif query in b:
#      print(b, onto_id)
# else:
# =print('match not found')
# matches plant embryo proper PO:0000001; plant embryo PO:0009009; microsporangium PO:0025202; microsporangium PO:0025202 (#it didn't find all the occurances of the
# terms nor microsporangium wall as a bigram.


# from this we learn that for the match to be made, we need to join the bigrams to pair of two words not separated by ','

# for word1, word2 in list_of_bigrams_testing:
# list_compression = word1 + word2
# print(list_compression)
print(TreebankWordDetokenizer().detokenize(['the', 'quick', 'brown']))

# join = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
# join = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in list_of_bigrams_testing]).strip()

# iterating over the list of tuples and joining the words

combined_bigram = []
for p in list_of_bigrams_testing:
    # microsporangium wall #whole plant
    combined_bigram.append('{} {}'.format(p[0], p[1]))
print('this is combined bigram', combined_bigram)  # combined bigram = ['microsporangium wall', 'whole plant']

for a in combined_bigram:
    print('this is the first bigram in the tuple:', a)

for query1, onto_id in onto.items():
    if query1 in combined_bigram:
        print('found it', query1, onto_id)

# but want to be adding to have a final result combined_bigram = [('microspangiu wall'), ('whole plant')]
'''
bigrams=[('more', 'is'), ('is', 'said'), ('said', 'than'), ('than', 'done')]
for a, b in bigrams:
    print(a)
    print(b)
'''

sentence = 'this is a foo bar sentences and i want to ngramize it'

n = 6
sixgrams = ngrams(sentence.split(), n)

for grams in sixgrams:
    print(grams)

n = 3
threegrams = ngrams(text_article.split(), n)
# threegrams1 = list(ngrams(text_article.split(), n))

ok = []
for grams1 in threegrams:
    # print(grams1)
    ok.append(
        grams1)  # [('Kugler', 'et', 'al.'), ('et', 'al.', 'BMC'), ('al.', 'BMC', 'Genomics'), ('BMC', 'Genomics', '2013,'), ('Genomics', '2013,', '14:728')
    # so in order to find matches of onto dictionary and the 3grams, the 3grams must not be separated by comma. So need to join.
print(ok)

onto_items_dummy = {'name': 'id', 'plant embryo proper': 'PO:0000001', ('Kugler', 'et', 'al.'): 'hello'}
for query1, onto_id in onto_items_dummy.items():
    if query1 in ok:
        print('found it', query1, onto_id)
#  else:
#      print('didnt find a matching ontology term')

'''

    for query1, onto_id in onto.items():
         if query1 in grams1:
            print('found it', query1, onto_id)

    print('didnt find a match')
'''

# read text
# raw data is the PO ontology file

raw_data = open('plant-ontology-dev.txt').read()

parsed_data2 = raw_data.replace('\n', '\t').split('\t')
definition_list = parsed_data2[2::6]

synonym_list = parsed_data2[3::6]
print("ontology synonym list is", synonym_list[0:20])

# remove dashes ('-') and underscores ('_') from the synonym_list words.

import string

string.punctuation


def remove_punctuation(txt):
    txt_nopunt = [''.join(c for c in s if c not in string.punctuation) for s in txt]  # must do a join function
    return txt_nopunt


new_sentences = remove_punctuation(synonym_list)


# works but it makes leaf-derived into leafderived
# also I don't want to remove % symbol, so best use the replace method.


def replace_certain_punctuations(list):
    records = [rec.replace('-', ' ').replace('_', ' ').replace('.', ' ') for rec in list]
    return records


print('synonym_name_list1 without punctuations -_.', replace_certain_punctuations(synonym_list))


####################################
# read text, find and print the phrases which include the data_reproducibility_keywords.
# if any such data_reproducibility_keywords have been found, then add a point to the reproducibility metrics score

# how to calculate the reproducibility metrics score.
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
