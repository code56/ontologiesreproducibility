__author__ = 'Evanthia_Kaimaklioti Samota'

#!/usr/bin/env python
# coding: utf-8

# In[7]:

import pandas as pd
import os, os.path, pdftotext, re, string, nltk, argparse,sys, functools, operator
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pathlib import Path
import urllib.request, urllib.error, urllib.parse, urllib
from pandas import read_html
import html5lib
from bs4 import BeautifulSoup


url = 'https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-728'




#need to work with an html file format of the paper

def fromurltotext(url):
    response = urllib.request.urlopen(url)
    webContent = response.read()
    text_article = ''

    # save the HTML file locally
    f = open('paper.html', 'wb')
    f.write(webContent)

    # Python - How to read HTML line by line into a list
    with open("paper.html") as f:
        content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    # want to get the text from the HTML
    f1 = open('hello.txt', "w+")  # must make it so the name changes each time
    #text_article = f1.write(str(content))  # I dont want new lines so f.write("\n\n".join(pdf))
    text_article = f1.write("\n\n".join(content))  # I dont want new lines so f.write("\n\n".join(pdf))

    #print(text_article)
    f1.close()
    f.close()
    print('done writing to hello.txt')
    return(text_article)

    # print("this is new log")
    # print(new_log) #-this works

    # put this in a function that will read many articles, not just one.
    # add the code of converting the pdf to text


url = 'https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-728'
text = fromurltotext(url)

'''
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
            f.write(" ".join(pdf)) #I dont want new lines so f.write("\n\n".join(pdf))
            f.close()
        print('done writing to ' + '%s.txt'%item.name + ' file')

# have as separate functions
#start with the functions ('what do I want the code to do').
# open files in folder ; convert to text ; manipulate the text all separate functions
# with open('%s.txt'%item.name

'''

'''
with open('output.txt') as f:
    log = f.readlines()


text_article = ''
for line in log:
    text_article += line


# pre-processing of text_article data
# remove punctuation
'''


def nlp_text_processing(txt):

    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    stopwords = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()
    tokenised_data = nltk.word_tokenize(txt)  # parsed_data1 instead of text_article?
    #print('tokenised data', tokenised_data)
    return (tokenised_data)


with open ('hello.txt') as f:
    log = f.readlines()
    text_article = ''
    for line in log:
        text_article += line

#print('this is text_article', text_article)

nlp_processed_text = nlp_text_processing(text_article) #function passes/works

#But need to get rid of div, and < > and /html and all the other html markup

''' 
def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


parsed_data1 = remove_punctuation(text_article)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


tokenised_data = nltk.word_tokenize(text_article) #parsed_data1 instead of text_article?
print('tokenised data', tokenised_data)

'''

bigram = list(ngrams(nlp_processed_text, 2))


#bigram = list(ngrams(tokenised_data, 2))
print('this is bigram before removing punctuation from article text', bigram)

'''
#example
word_data = "The best performance can bring in sky high success."
nltk_tokens = nltk.word_tokenize(word_data)
list_of_bigrams = list(nltk.bigrams(nltk_tokens))
print(list_of_bigrams)
'''

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
print('this is tokens', tokens[:100]) #list structure

# Filter out stop words


# remove remaining tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
# filter out stop words

words = [w for w in tokens if not w in stopwords]
print('this is words after removing stop words', words[:100])

# https://machinelearningmastery.com/clean-text-machine-learning-python/


#works

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+')

'''
cv = CountVectorizer(lowercase=True, ngram_range=(1, 4), tokenizer=token.tokenize)

X = cv.fit(words)
#print(X.vocubulary_)
print('am printing cv.get_feature_names')
print(cv.get_feature_names())

X = cv.transform(words)
# X = cv.fit_transform(words)

#what is the purpose of these?
#print("X.shape", X.shape)
#print("X", X)
#print("X.toarray()", X.toarray())


df = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
#print('df', df)
'''

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

#create a dictionary with keys the name of the plant ontology and value the PO term from the plant-onto-dev.txt.
	if len(split) > 2:
		onto[split[1]] = split[0]
print('plant ontology dictionary', onto)

'''
f = open("dictionary_out.txt", "a")
print('this is ontologies dictionary', file=f)
print(onto, file=f)
f.close()
'''


#would be good to be searching for terms in whole text except in the references.


#this only returns one word matches for keys, e.g. found query lemma PO:0009037
for query, onto_id in onto.items():
	if query in tokenised_data:
		print('found single word matches', query, onto_id)


'''
for query, onto_id in onto.items():
    wordlist = re.sub("[^\w]"," ", query).split()
    for word in wordlist:
        if word in words: #words is the tokenised journal article
            print('found the word in manuscript:', word + " | " + 'query: ' + query + " | " + 'onto_id: ' + onto_id)

#this results in a lot of false positives
#found the word in manuscript: whole | query: whole plant fruit formation stage 70% to final size | onto_id: PO:0007027
'''

#match the bigrams and the keys in the onto dictionary

# joining tuple elements
# using join() + list comprehension

res = [' '.join(tups) for tups in bigram]
print("The joined data res is : " + str(res))

'''
for query, onto_id in onto.items():
    if query in res:
        print('found query match in bigrams', query)
'''

#lateral root, vs lateral root tip, vs all occurances of keys with the word lateqral in it
string = 'plant embryo proper'
arr = [x.strip() for x in string.strip('[]').split(' ')]
#result -> ['plant', 'embryo', 'proper']

'''
mystr = 'This is a string, with words!'
wordList = re.sub("[^\w]", " ",  mystr).split()
'''



#This is the wall of the microsporangium.
#vs I found in my garden wall a microsporangium (not wanted).

list_of_bigrams_testing = [('microsporangium', 'wall'), ('whole', 'plant')]
for query, onto_id in onto.items():
    res = [' '.join(tups) for tups in list_of_bigrams_testing]
    if query in res:
        print('found query in the res', query, onto_id)
print('this is res tuple', res)
        #if query in a:
        #    print(a, onto_id)
       # elif query in b:
      #      print(b, onto_id)
    #else:
        #=print('match not found')
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
    if query1 in combined_bigram:
        print('found it', query1, onto_id)

'''
for query, onto_id in onto.items():
	if query in text:
		print(query, onto_id)
'''

#but want to be adding to have a final result combined_bigram = [('microspangiu wall'), ('whole plant')]
'''
bigrams=[('more', 'is'), ('is', 'said'), ('said', 'than'), ('than', 'done')]
for a, b in bigrams:
    print(a)
    print(b)
'''

#read text
#raw data is the PO ontology file

raw_data = open('plant-ontology-dev.txt').read()

parsed_data2 = raw_data.replace('\n','\t').split('\t')
definition_list = parsed_data2[2::6]

synonym_list = parsed_data2[3::6]
print("ontology synonym list is", synonym_list[0:20])

#remove dashes ('-') and underscores ('_') from the synonym_list words.

import string
string.punctuation

def remove_punctuation(txt):
    txt_nopunt = [''.join(c for c in s if c not in string.punctuation) for s in txt] #must do a join function
    return txt_nopunt

new_sentences = remove_punctuation(synonym_list)

#works but it makes leaf-derived into leafderived
#also I don't want to remove % symbol, so best use the replace method.


def replace_certain_punctuations(list):
    records = [rec.replace('-', ' ').replace('_', ' ').replace('.', ' ') for rec in list]
    return records

print('synonym_name_list1 without punctuations -_.', replace_certain_punctuations(synonym_list))



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
    concordance_txt = ([text.tokens[list(map(lambda x: x - 5 if (x - left_margin) > 0 else 0, [offset]))[0]:offset + right_margin]
     for offset in c.offsets(target_word)])

    ## join the sentences for each of the target phrase and return it
    return [''.join([x + ' ' for x in con_sub]) for con_sub in concordance_txt]


data_reproducibility_keywords = ['accession', 'data', 'Supporting', 'available', 'repository', 'GO', 'EBI', 'ArrayExpress', 'PO', 'sequences', 'expression', 'snps', 'genes', 'wheat', 'rice']
phrases_from_article = []
for word in data_reproducibility_keywords:
    phrases_from_article = get_all_phrases_containing_tar_wrd(word, text_article)
    print('phrases from text article', word, phrases_from_article) #it doesn't return all the occurences or the complete
                                                                   # sentences because it is two column text.

#returning sentences containing particular phrases: e.g. "Supporting data"
def regex_search(filename, term):
    searcher = re.compile(term + r'([^\w-]|$)').search
    with open(filename, 'r') as source, open("new.txt", 'w') as destination:
        for line in source:
            if searcher(line):
                destination.write(line) #fclose?

find_phrases = regex_search('ontopaper_usecase.pdf.txt', 'accession number')


#return complete sentences if they contain 'wanted' word.
txt = "I like to eat apple. Me too. Let's go buy some apples."
hello = [sentence + '.' for sentence in txt.split('.') if 'apple' in sentence]
print(hello)
#['I like to eat apple.', " Let's go buy some apples."]

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
    data = file.read().replace('\n', '').replace('\t', '') #parsed_data2 = raw_data.replace('\n','\t').split('\t')
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


#print(remove_newlines("ontopaper_usecase.pdf.txt"))

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

