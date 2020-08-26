#!/usr/bin/env python
# coding: utf-8

# In[7]:

import pandas as pd
import re
import string
import nltk, argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

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

#working on the PO ontology development file
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


text = 'asdfsdkjfhskjh, plant embryo proper'

for query, onto_id in onto.items():
	if query in text:
		print(query, onto_id)

#for bigram (i.e.token pair), onto_id in onto.items:
    #if bigram in list_of_bigrams:
      #  print(bigram, onto_id)
list_of_bigrams_testing = [('microsporangium', 'wall'), ('whole', 'plant')]
for query1, onto_id in onto.items():
    for a, b in list_of_bigrams_testing:
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
for p in list_of_bigrams_testing:
    combined_bigram = [('{} {}'.format(p[0], p[1]))]
    print(combined_bigram)
#['microsporangium wall']
#['whole plant']

for a, b in combined_bigram:
    print('this is the first bigram in the tuple', a)
    print(b)
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

