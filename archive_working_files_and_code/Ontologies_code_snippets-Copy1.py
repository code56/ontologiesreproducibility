#!/usr/bin/env python
# coding: utf-8

# In[7]:

import pandas as pd
import re
import string
import nltk

with open('output.txt') as f:
	log = f.readlines()

text_article = ''
for line in log:
	text_article += line
    
#print(text_article)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
print('this is stopwords', stopwords)
print(ps)

tokenised_data = nltk.word_tokenize(text_article)
#print(tokenised_data)

#for i in word_tokenize(raw_data):
    #print (i)

parsed_data = text_article.replace('\t', '\n').split('\n')
parsed_data[0:10]
#type(parsed_data)

#print(parsed_data)

print(string.punctuation)
#remove punctuation

def remove_punctuation(txt):
    txt_nopunt = [c for c in txt if c not in string.punctuation]
    return txt_nopunt

parsed_data1 = remove_punctuation(parsed_data)
#print('this is parsed_data1', parsed_data1)

parsed_data2 = remove_punctuation(tokenised_data)
#print('this is parsed_data2 tokenised with removed punctuation', parsed_data2)

#stop_words = set(stopwords.words('english'))

#word_tokenize() function splits strings into tokens (normally words)
#based on white space and punctuation (commas and periods are taken as 
#separate tokens)

words = nltk.word_tokenize(text_article)
#print('this is words', words)

#words1 = word_tokenize(parsed_data2) #word_tokenize expects string. 
#print('this is stop words removed from the data', words1)

#Count Vectorization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2))

X = cv.fit(parsed_data1)

print('this is X', X) #note see later to what data should make this on type - string, corpus, and which parsed_data etc..

print(X.vocabulary_)


def tokenize(txt):
    re.split('\W+', txt)
    return tokens


#text cleaning
def clean_text(txt):
    txt = "".join([c for c in txt if c not in string.punctuation])
    tokens = re.split('\W+', txt)
    txt = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return txt


clean_text(parsed_data)

with open('output.txt') as f:
	log = f.readlines()

text_article = ''
for line in log:
    text_article += line

print(string.punctuation)  # printing the punctuation values


def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct
 
parsed_data1 = remove_punctuation(text_article)
#print(parsed_data1)

#Text cleaning: remove stop words.
#stopwords = nltk.corpus.stopwords.words('english')
#ps = nltk.PorterStemmer()

'''
def remove_punctuation(txt):
    txt_nopunt = "".join([c for c in txt if c not in string.punctuation])
    tokens = re.split('\W+', txt)
    txt = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return txt_nopunt
''' 

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


from nltk.tokenize import word_tokenize
tokens = word_tokenize(parsed_data1)
print('this is tokens', tokens[:100])


#Filter out stop words

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print('this is stop_words', stop_words)

# remove remaining tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
# filter out stop words

words = [w for w in tokens if not w in stop_words]
print('this is words', words[:100])

#https://machinelearningmastery.com/clean-text-machine-learning-python/


# In[ ]:


#doesnt works all well#

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+')

cv = CountVectorizer(lowercase=True, ngram_range = (1,3),tokenizer = token.tokenize)

text_counts = cv.fit(words)

txt1 = cv.get_feature_names()
#print('this is text counts', text_counts)

#print(txt1)

v_1 = cv.transform(txt1)
print(v_1.shape)
print(v_1)
print(v_1.toarray())


# In[ ]:


#good and works

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, ngram_range = (1,2),tokenizer = token.tokenize)
text_counts= cv.fit_transform(words)
print(text_counts)


# In[ ]:


#works


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+')

cv = CountVectorizer(lowercase=True, ngram_range = (1,2),tokenizer = token.tokenize)

X = cv.fit(words)
#print(X.vocubulary_)
print(cv.get_feature_names())

X = cv.transform(words)
# X = cv.fit_transform(words)

print(X.shape)
print(X)
print(X.toarray())


df = pd.DataFrame(X.toarray(), columns = cv.get_feature_names())
print(df)


# In[ ]:


nltk.download('genesis')
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
finder.nbest(bigram_measures.pmi, 10)  # doctest: +NORMALIZE_WHITESPACE
[(u'Allon', u'Bacuth'), (u'Ashteroth', u'Karnaim'), (u'Ben', u'Ammi'),
 (u'En', u'Mishpat'), (u'Jegar', u'Sahadutha'), (u'Salt', u'Sea'),
 (u'Whoever', u'sheds'), (u'appoint', u'overseers'), (u'aromatic', u'resin'),
 (u'cutting', u'instrument')]


# In[ ]:


#make bigrams from the words 

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(2,3))
#will look for bigram, trigrams

corpus = ['This is a sentence is', 'This is another sentence', 'third document is here']

X = cv.fit(corpus)
print(X.vocabulary_)
print(cv.get_feature_names())

X = cv.transform(corpus)
X = cv.fit_transform(corpus)
print('this is number of unique ', X.shape)
print('this is x', X)
print('this is x.array', X.toarray())

df = pd.DataFrame(X.toarray(), columns = cv.get_feature_names())
print(df)


cv1 = CountVectorizer(ngram_range=(2,2))
x = cv1.fit_transform(words)
print(X.shape)

print(cv1.get_feature_names())


filtered_data = [w for w in words if not w in stop_words]
print(filtered_data)


'''
filtered_data = []
for w in words:
    if w not in stop_words:
        filtered_data.append(w)
print(filtered_data)
'''


# In[ ]:


#remove punctuation
#also works 
import string
string.punctuation

def remove_punctuation(txt):
    txt_nopunt = [c for c in txt if c not in string.punctuation]
    return txt_nopunt

new_sentences = remove_punctuation(filtered_data)
print(new_sentences)
type(new_sentences)


# In[ ]:


#stemming: data pre-processing. Take the root stem of the word
# but stemming is not very accurate, but it's fast.
# do I need to do stemming in order to get what I need? 

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']

#for w in example_words:
 #   print(ps.stem(w))
    
#for w in new_sentences:
  #  print(ps.stem(w))
    

def stem_the_words(txt):
    new_words = [ps.stem(w) for w in txt]
    for w in txt:
        print('I have done the stemming!')
        return new_words
    
new_words = stem_the_words(new_sentences)
print(new_words)


# In[ ]:


def remove_punctuation(txt):
    txt_nopunt = [c for c in txt if c not in string.punctuation]
    return txt_nopunt



#examples of keywords in the onto {} (Plant ontologies dictionary) which have punctuation
#'root-derived cultured plant cell': 'PO:0000008'

#create a dictionary with keys the name of the plant ontology and value the PO term. 
onto = {}
for line in open('plant-ontology-dev.txt'):
	split = line.split("\t")
print(split)
    #if len(split) > 2:
#		onto[split[1]] = split[0]
#print('this is the Plant Ontologies constructed dictionary', onto)

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

#>>> x = [''.join(c for c in s if c not in string.punctuation) for s in x]
#>>> print(x)
#['hello', '', 'h3a', 'ds4']
#print(string.punctuation)

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
#so replaace all symbols in the items with a space. 
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


# In[ ]:


def replace_certain_punctuations(list):
    records = [rec.replace("-", " ").replace('_', ' ').replace('.', ' ') for rec in list]
    return records

name_list1 = replace_certain_punctuations(name_list)
print(name_list1)


# In[ ]:



# Python3 code to demonstrate  
# conversion of lists to dictionary 
# using dictionary comprehension 
  
# initializing lists 
test_keys = ["Rash", "Kil", "Varsha"] 
test_values = [1, 4, 5] 
  
# Printing original keys-value lists 
print ("Original key list is : " + str(test_keys)) 
print ("Original value list is : " + str(test_values)) 
  
# using dictionary comprehension 
# to convert lists to dictionary 
res = {test_keys[i]: test_values[i] for i in range(len(test_keys))} 
  
# Printing resultant dictionary  
print ("Resultant dictionary is : " +  str(res)) 


# In[ ]:



# Python3 code to demonstrate  
# conversion of lists to dictionary 
# using dictionary comprehension 

  
# using dictionary comprehension 
# to convert lists to dictionary 
onto = {name_list1[i]: id_list[i] for i in range(len(name_list1))} 
  
# Printing resultant dictionary  
print ("Resultant dictionary is : " +  str(onto)) 


# In[ ]:



onto = {}

for line in open('plant-ontology-dev.txt'):
	split = line.split("\t")

#create a dictionary with keys the name of the plant ontology and value the PO term. 
	if len(split) > 2:
		onto[split[1]] = split[0]
print('this is the Plant Ontologies constructed dictionary', onto)

import re

#the matching done here is wrong, as we need to be matching complete words and not characters, this is where NLP will help.
# can also query the article simultaneously for data availability related keywords, such as: 'data availability', 'data is available at',
occurrences = {}
for key in onto:
    m = re.findall(key, text_article, re.IGNORECASE)
    occurrences[key] = len(m)
print('this is the occurrences', occurrences)


#must split the article document text into words otherwise, it might find matches from within text
#for example instead of matching word "flowering" it matches just "flower"
# OR finds words where they are not actual words, e.g. "sySTEM" finds "stem"
import re
sentence = "Dogs are great pets and hamsters are bad pets. That is why I want a dog"

scores = {'dog' : 5, 'hamster' : -2}

occurrences = {}

for key in scores:
  m = re.findall(key, sentence , re.IGNORECASE)
  occurrences[key] = len(m)

totalScore = 0

#lower() built-in method, converts all uppercase characters into lowercase
for word in occurrences:
  totalScore += scores.get(word.lower(), 0) * occurrences[word]

print(totalScore)

#end of testing code for words
#########




#but this is wrong as it will confuse 'stem' with 'system'.
for query, onto_id in onto.items():
	if query in text_article:
		print(query, onto_id)  #print the key found in the string of text and its pair from the dictionary 


# In[ ]:


print(parsed_data)


# In[ ]:


import re
for query, onto_id in onto.items():
	if query in words:
		print(query, onto_id)  #print the key found in the string of text and its pair from the dictionary 


# In[ ]:


#create a data structure to keep all 'key words, or key phrases' such 'data available at'
# the use of k-mers and n-grams will help? or can Python match complete elements?
# it would be helpful to clean the text to stemming? So that plural can be also caught? 
# what about things like regex to catch accession numbers such as: E-MTAB-1729.

keywords_list = ['data', 'data available at', 'data sets', 'data set', 'data sets are available in', 'data sets are available at', 'EBI', 'EBI ArrayExpress', 'repository', 'accession number', 'supporting data', 'Supporting data', 'data set', 'ArrayExpress repository', 'wwww.ebi.ac.uk/arrayexpress', 'data is available as', 'supplemental online material ']
print('this is keywords_list', keywords_list)


# In[ ]:


## create a filter that consults a scored dictionary and a list of query words, from http://stackoverflow.com/questions/23479179/combining-filters-in-nltk-collocations
def create_bigram_filter_minfreq_inwords(scored, words, minfreq):
    def bigram_filter(w1, w2):
        return (w1 not in words and w2 not in words) and (
                (w1, w2) in scored and scored[w1, w2] <= minfreq)

    return bigram_filter


def create_trigram_filter_minfreq_inwords(scored, words, minfreq):
    def trigram_filter(w1, w2, w3):
        return (w1 not in words and w2 not in words and w3 not in words) and (
                (w1, w3, w3) in scored and scored[w1, w2, w3] <= minfreq)

    return trigram_filter


# In[ ]:

import nltk
 
def get_all_phrases_containing_tar_wrd(target_word, tar_passage, left_margin = 10, right_margin = 10):
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
    c = nltk.ConcordanceIndex(text.tokens, key = lambda s: s.lower())
 
    ## Collect the range of the words that is within the target word by using text.tokens[start;end].
    ## The map function is use so that when the offset position - the target range < 0, it will be default to zero
    concordance_txt = ([text.tokens[map(lambda x: x-5 if (x-left_margin)>0 else 0,[offset])[0]:offset+right_margin] for offset in c.offsets(target_word)])
                         
    ## join the sentences for each of the target phrase and return it
    return [''.join([x+' ' for x in con_sub]) for con_sub in concordance_txt]
 
## Test the function
 
## sample text from http://www.shol.com/agita/pigs.htm
raw  = """The little pig saw the wolf climb up on the roof and lit a roaring fire in the fireplace and          placed on it a large kettle of water.When the wolf finally found the hole in the chimney he crawled down          and KERSPLASH right into that kettle of water and that was the end of his troubles with the big bad wolf.          The next day the little pig invited his mother over . She said &amp;amp;quot;You see it is just as I told you.           The way to get along in the world is to do things as well as you can.&amp;amp;quot; Fortunately for that little pig,          he learned that lesson. And he just lived happily ever after!"""
 
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)
text.concordance('wolf') # default text.concordance output
 
## output:
## Displaying 2 of 2 matches:
##                                     wolf climb up on the roof and lit a roari
## it a large kettle of water.When the wolf finally found the hole in the chimne
 

print ('Results from function')
results = get_all_phrases_containing_tar_wrd('wolf', raw)
for result in results:
    print(result)
 
## output:
## Results from function
## The little pig saw the wolf climb up on the roof and lit a roaring
## large kettle of water.When the wolf finally found the hole in the chimney he crawled


# In[ ]:

    concordance_txt = ([text.tokens[map(lambda x: x-5 if (x-left_margin)>0 else 0,[offset])[0]:offset+right_margin] for offset in c.offsets(target_word)])

'''    
    April 26, 2017 at 5:08 pm

    Great function! For python 3, we would need to change the syntax slightly. I believe that the map function in version 3 returns an iterator instead of a list – we simply need to make “map” a list.

    list(map(lambda x: x-5 if (x-left_margin)>0 else 0,[offset]))
'''

# It does work 🙂 thank you for the function! Since nobody pointed it out so far: there is typo in the code: the function is defined as ‘get_all_phases_containing_tar_wrd’ (line 3) with an ‘r’ missing in phRases but call with ‘get_all_phrases_containing_tar_wrd’ (line 51) with the ‘r’




# In[ ]:



## collect word phases, from https://simply-python.com/2014/03/14/saving-output-of-nltk-text-concordance/
def word_phases(target_word, query_text, left_margin=10, right_margin=10):
    """
        Function to get all the phases that contain the target word in a text/passage tar_passage.

        str target_word, str tar_passage int left_margin int right_margin --> list of str
        left_margin and right_margin allocate the number of words/pununciation before and after target word
        Left margin will take note of the beginning of the text
    """
    ## Collect all the index or offset position of the target word
    c = nltk.ConcordanceIndex(query_text.tokens, key=lambda s: s.lower())

    ## Collect the range of the words that is within the target word by using text.tokens[start;end].
    ## The map function is use so that when the offset position - the target range < 0, it will be default to zero
    concordance_txt = (
    [query_text.tokens[list(map(lambda x: x - 5 if (x - left_margin) > 0 else 0, [offset]))[0]:offset + right_margin]
     for offset in c.offsets(target_word)])

    ## join the sentences for each of the target phrase and return it
    return [''.join([x + ' ' for x in con_sub]) for con_sub in concordance_txt]


# In[ ]:


def extract_phases(tokens, wordlist):
    all_phases = []
    text = nltk.Text(tokens)
    for word in wordlist:
        phases = word_phases(word, text)
        if phases:
            all_phases.append(phases)
    return all_phases


# In[ ]:



def find_and_filter_bigrams(tokens):
    # initialize finder object with the tokens
    finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

    # build a dictionary with bigrams and their frequencies
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    scored = dict(finder.score_ngrams(bigram_measures.raw_freq))

    # create the filter...
    myfilter = create_bigram_filter_minfreq_inwords(scored, wordlist, 0.1)
    #    before_filter = list(finder.score_ngrams(bigram_measures.raw_freq))
    #    if args.verbose:
    #        print('Before filter:\n', before_filter)

    # apply filter
    finder.apply_ngram_filter(myfilter)

    # remove empty
    finder.apply_word_filter(lambda w: w in (''))

    # remove low freq
    finder.apply_freq_filter(2)

    # after_filter = list(finder.score_ngrams(bigram_measures.raw_freq))
    best_filter = sorted(finder.nbest(bigram_measures.raw_freq, 5))
    if args.verbose:
        # print('\nAfter filter:\n', after_filter)
        print('\nTop bigrams:\n', best_filter)
    return best_filter


def find_and_filter_trigrams(tokens):
    # initialize finder object with the tokens
    finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)

    # build a dictionary with bigrams and their frequencies
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    scored = dict(finder.score_ngrams(trigram_measures.raw_freq))

    # create the filter...
    myfilter = create_trigram_filter_minfreq_inwords(scored, wordlist, 0.1)

    # apply filter
    finder.apply_ngram_filter(myfilter)

    # remove empty
    finder.apply_word_filter(lambda w: w in (''))

    # remove low freq
    finder.apply_freq_filter(2)

    # after_filter = list(finder.score_ngrams(trigram_measures.raw_freq))
    best_filter = sorted(finder.nbest(trigram_measures.raw_freq, 5))
    if args.verbose:
        # print('\nAfter filter:\n', after_filter)
        print('\nTop trigrams:\n', best_filter)
    return best_filter


# In[ ]:



def score_chunks(chunkparser, tree, trace=0):
    chunkscore = chunk.ChunkScore()

    tokens = tree.leaves()
    gold = chunkparser.parse(Tree('S', tokens), trace)
    chunkscore.score(gold, tree)

    a = chunkscore.accuracy() * 100
    p = chunkscore.precision() * 100
    r = chunkscore.recall() * 100
    f = chunkscore.f_measure() * 100
    i = chunkscore.incorrect()
    m = chunkscore.missed()

    if args.verbose:
        print('Testing: {0}'.format(tree))
        print('Guessed: {0}'.format(chunkscore.guessed()))
        print('\n/' + ('=' * 75) + '\\')
        print('Scoring', chunkparser)
        print(('-' * 77))
        print('Accuracy: %5.1f%%' % (a), ' ' * 4, end=' ')
        print('Precision: %5.1f%%' % (p), ' ' * 4, end=' ')
        print('Recall: %5.1f%%' % (r), ' ' * 6, end=' ')
        print('F-Measure: %5.1f%%' % (f))
        print('Missing: {0} -> {1}'.format(len(m), m))
        print('Incorrect: {0} -> {1}'.format(len(i), i))

    return (a, p, r, f, i, m)


# In[ ]:


# Python3 code to demonstrate 
# Bigram formation 
# using list comprehension + enumerate() + split() 
   
# initializing list  
test_list = ['geeksforgeeks is best', 'I love']
  
# printing the original list  
print ("The original list is : " + str(test_list))
# using list comprehension + enumerate() + split() 
# for Bigram formation 
res = [(x, i.split()[j + 1]) for i in test_list for j, x in enumerate(i.split()) if j < len(i.split()) - 1]
  
# printing result
print ("The formed bigrams are : " + str(res)) 


# In[ ]:


from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

textWords = [w.lower() for w in webtext.words('pirates.txt')]

finder = BigramCollocationFinder.from_words(textWords)
finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)


# In[ ]:


import nltk
with open('sample_text.txt', 'r') as f:
    sample_text = f.read()
    
import regex as re 
'''
def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[*\w\s]', '', word)
        if new_word != word;

'''
# In[ ]:


#the purpose of working with bigrams is because in the PO_dict, some keys are bigrams
#so I have to make the article_text into bigrams too and then see if there are any matches

#bigrm = list(nltk.bigrams(text.split()))

import nltk
bigrm = nltk.bigrams(words)
print(bigrm)
    
#>>> import nltk
#>>> from nltk.tokenize import word_tokenize
#>>> text = "to be or not to be"
#>>> tokens = nltk.word_tokenize(text)
#>>> bigrm = nltk.bigrams(tokens)
#>>> print(*map(' '.join, bigrm), sep=', ')
#to be, be or, or not, not to, to be    
import nltk
from nltk.tokenize import word_tokenize
tokens = nltk.word_tokenize(words)
bigrm = nltk.bigrams(tokens)
#print(*map(' '.join, bigrm), sep=', ')


# In[ ]:




