__author__ = 'Evanthia_Kaimaklioti Samota'

#!/usr/bin/env python
# coding: utf-8

# In[7]:

import pandas as pd
import os, os.path, pdftotext, re, string, nltk, argparse,sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pathlib import Path

#put this in a function that will read many articles, not just one.
#add the code of converting the pdf to text

yourpath = os.getcwd()

data_folder = Path("papers_collection_folder") #can be changed to user input data
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
            f = open('%s.txt'% item.name, "w+") #must make it so the name changes each time
            f.write("\n\n".join(pdf))
            f.close()
        print('done writing to ' + '%s.txt'%item.name + ' file')

#should I put this all inside the loop above?
# with open('%s.txt'%item.name

with open('output.txt') as f:
    log = f.readlines()

text_article = ''
for line in log:
    text_article += line

# pre-processing of text_article data
# remove punctuation


def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


parsed_data1 = remove_punctuation(text_article)
# print(parsed_data1)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

#should i have applied the removal of punctuation first before tokenisation?

tokenised_data = nltk.word_tokenize(text_article) #parsed_data1 instead of text_article?
print('tokenised data', tokenised_data)

#aim convert tokenized data which is  a list into a string
# for i in word_tokenize(raw_data):
# print (i)

# Text cleaning: remove stop words.
# stopwords = nltk.corpus.stopwords.words('english')
# ps = nltk.PorterStemmer()

'''
def remove_punctuation(txt):
    txt_nopunt = "".join([c for c in txt if c not in string.punctuation])
    tokens = re.split('\W+', txt)
    txt = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return txt_nopunt
'''

tokens = word_tokenize(parsed_data1)
print('this is tokens', tokens[:100])

# Filter out stop words


# remove remaining tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
# filter out stop words

words = [w for w in tokens if not w in stopwords]
print('this is words', words[:100])

# https://machinelearningmastery.com/clean-text-machine-learning-python/


#works

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+')

cv = CountVectorizer(lowercase=True, ngram_range=(1, 4), tokenizer=token.tokenize)

X = cv.fit(words)
#print(X.vocubulary_)
print('am printing cv.get_feature_names')
print(cv.get_feature_names())

X = cv.transform(words)
# X = cv.fit_transform(words)


print("X.shape", X.shape)
print("X", X)
print("X.toarray()", X.toarray())


df = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
print('df', df)

#example
word_data = "The best performance can bring in sky high success."
nltk_tokens = nltk.word_tokenize(word_data)
list_of_bigrams = list(nltk.bigrams(nltk_tokens))
print(list_of_bigrams)

#compare a list_of_bigrams with the plant ontology dictionary. However, the keys in the onto{} are not bigrams -
# i.e not two tokens. Then how do I do the comparison by ignoring the comma? Would I have to make a new onto dictionary
# with tokens separated by coma for the keys? or can I make a new list of bigrams NOT separated by commas?

#working on the plant-ontology-dev.txt
#examples of keywords in the onto {} (Plant ontologies dictionary) which have punctuation
#'root-derived cultured plant cell': 'PO:0000008'
#create a dictionary with keys the name of the plant ontology and value the PO term.

onto = {}
for line in open('plant-ontology-dev.txt'):
	split = line.split("\t")

#create a dictionary with keys the name of the plant ontology and value the PO term.
	if len(split) > 2:
		onto[split[1]] = split[0]
print (onto)
'''
f = open("dictionary_out.txt", "a")
print('this is ontologies dictionary', file=f)
print(onto, file=f)
f.close()
'''

#this only returns one word matches for keys, e.g. cotyledon: PO:0020030
for query, onto_id in onto.items():
	if query in text_article:
		print('found query', query, onto_id)


for query, onto_id in onto.items():
    wordlist = re.sub("[^\w]"," ", query).split()
    for word in wordlist:
        if word in words: #tokenised journal article
            print('found the word in manuscript:', word + " | " + 'query: ' + query + " | " + 'onto_id: ' + onto_id)


#lateral root, vs lateral root tip, vs all occurances of keys with the word lateral in it
string = 'plant embryo proper'
arr = [x.strip() for x in string.strip('[]').split(' ')]
#result -> ['plant', 'embryo', 'proper']

'''
mystr = 'This is a string, with words!'
wordList = re.sub("[^\w]", " ",  mystr).split()
'''

#for bigram (i.e.token pair), onto_id in onto.items:
    #if bigram in list_of_bigrams:
      #  print(bigram, onto_id)
list_of_bigrams_testing = [('microsporangium', 'wall'), ('whole', 'plant')]
for query1, onto_id in onto.items():
    for a, b in list_of_bigrams_testing:
        print(a,b)
        if query1 in a:
            print(a, onto_id)
        elif query1 in b:
            print(b, onto_id)
else:
    print('match not found')
#matches plant embryo proper PO:0000001; plant embryo PO:0009009; microsporangium PO:0025202; microsporangium PO:0025202 (#it didn't find all the occurances of the
#terms nor microsporangium wall as a bigram.


#from this we learn that for the match to be made, we need to join the bigrams to pair of two words not separated by ','

#for word1, word2 in list_of_bigrams_testing:
    #list_compression = word1 + word2
    #print(list_compression)
print(TreebankWordDetokenizer().detokenize(['the', 'quick', 'brown']))

#join = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
#join = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in list_of_bigrams_testing]).strip()

#iterating over the list of tuples and joining the words
for p in list_of_bigrams_testing:
    print('{} {}'.format(p[0], p[1]))
#microsporangium wall
#whole plant

combined_bigram = []
for p in list_of_bigrams_testing:  #microsporangium wall #whole plant
    combined_bigram = [('{} {}'.format(p[0], p[1]))]
    print('this is combined bigram', combined_bigram)
    print('this is p in combined_bigram', p)
#['microsporangium wall']
#['whole plant']

for a in combined_bigram:
    print('this is the first bigram in the tuple', a)
    #print(b)
'''
for query1, onto_id in onto.items():
    for a in combined_bigram:
        if query1 in a:
            print(a, onto_id)
        else:
            print('combined bigram list match not found')
'''

for query1, onto_id in onto.items():
    for query1 in combined_bigram:
        print(query1, onto_id)
else:
    print('combined bigram list match not found')

for query1, onto_id in onto.items():
    if query1 in combined_bigram:
        print('found it', query1, onto_id)

'''
for query, onto_id in onto.items():
	if query in text:
		print(query, onto_id)
'''

#but want to be adding to have a final result combined_bigram = [('microspangiu wall'), ('whole plant')]

bigrams=[('more', 'is'), ('is', 'said'), ('said', 'than'), ('than', 'done')]
for a, b in bigrams:
    print(a)
    print(b)

#read text
#raw data is the PO ontology file
raw_data = open('plant-ontology-dev.txt').read()
raw_data[0:500]

parsed_data2 = raw_data.replace('\n','\t').split('\t')
parsed_data2[0:20]

id_list = parsed_data2[2::6]
print(id_list[0:20])

name_list = parsed_data2[3::6]
print(name_list[0:20])

#remove dashes ('-') and underscores ('_') from the name_list words.
import string
string.punctuation

def remove_punctuation(txt):
    txt_nopunt = [''.join(c for c in s if c not in string.punctuation) for s in txt] #must do a join function
    return txt_nopunt

new_sentences = remove_punctuation(name_list)
print(new_sentences)
#type(new_sentences)
#works but it makes leaf-derived into leafderived
#also I don't want to remove % symbol, so best use the replace method.
#replace

#new_name_list = name_list.replace('-', ' ')
#print(new_name_list)

def replace_certain_punctuations(list):
    #items = ['_', '-', '.']
    records = [rec.replace("-", " ") for rec in list]
    return records

replace_certain_punctuations(name_list)

#this works to replace the dash with space.
#but we also want to apply this for replacing the other symbols in the items.
#so replace all symbols in the items with a space.
#do we iterate the items? and ask if found character in items replace with space?
def replace_these_punctuations(list):
    items = ['-','_']
    for item in items:
        records = [rec.replace(item, " ") for rec in list]
        return records

replace_these_punctuations(name_list)

# txt_nopunt = [''.join(c for c in s if c not in string.punctuation) for s in txt] #must do a join function

#this works to replace the _ with space.
#but doesn't work to replace the -

name_list1 = ['name', 'plant_embryo proper', 'anther-wall', 'gynoecium.primordium']


def replace_certain_punctuations(list):
    records = [rec.replace("-", " ").replace('_', ' ').replace('.', ' ') for rec in list]
    return records

replace_certain_punctuations(name_list1)

def replace_certain_punctuations(list):
    records = [rec.replace("-", " ").replace('_', ' ').replace('.', ' ') for rec in list]
    return records

name_list1 = replace_certain_punctuations(name_list)
print('name_list1', name_list1)

####################################
#read text, find and print the phrases which include the data_reproducibility_keywords.
#if any such data_reproducibility_keywords have been found, then add a point to the reproducibility metrics score

# how to calculate the reproducibility metrics score.
# add one point each time in the manuscript is found: data_reproducibility_keywords,

#from https://simply-python.com/2014/03/14/saving-output-of-nltk-text-concordance/

def get_all_phrases_containing_tar_wrd(target_word, tar_passage, left_margin=10, right_margin=10):
    """
        Function to get all the phases that contain the target word in a text/passage tar_passage.
        Workaround to save the output given by nltk Concordance function

        str target_word, str tar_passage int left_margin int right_margin --> list of str
        left_margin and right_margin allocate the number of words/pununciation before and after target word
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
    concordance_txt = ([text.tokens[list(map(lambda x: x - 5 if (x - left_margin) > 0 else 0, [offset]))[0]:offset + right_margin]
     for offset in c.offsets(target_word)])

    ## join the sentences for each of the target phrase and return it
    return [''.join([x + ' ' for x in con_sub]) for con_sub in concordance_txt]


data_reproducibility_keywords = ['accession', 'data', 'available', 'repository', 'GO', 'EBI', 'ArrayExpress', 'PO', 'sequences', 'expression', 'snps', 'genes', 'wheat', 'rice']
phrases_from_article = []
for word in data_reproducibility_keywords:
    phrases_from_article = get_all_phrases_containing_tar_wrd(word, text_article)
    print('phrases from text article', word, phrases_from_article)


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