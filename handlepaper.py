#this python file is to handle the paper side of things. To apply NLTK to find matches of 'interesting' words that we are looking for.
# OBJECTIVE: find phrases/words "accession number E-MTAB-1729", "ENA-ERP003465", "Study PRJEB4202",
# could be starting from the HTML, could be starting from the pdf.


import re, argparse, sys, scholarly, urllib, feedparser, nltk, bs4, string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

from nltk import word_tokenize, sent_tokenize, corpus
from nltk.text import Text
from nltk import chunk
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk.tree import Tree
from nltk.collocations import *
from nltk.corpus import wordnet, stopwords
from Bio import Entrez
from bs4 import BeautifulSoup

wnl = nltk.WordNetLemmatizer()

#input: HTML file
#objective: Find
def workwithhtml()

#function that creates a filter.
# create a filter that consults a scored dictionary and a list of query words, from http://stackoverflow.com/questions/23479179/combining-filters-in-nltk-collocations
# Input is:
#Objective is:
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



## collectword phases, from https://simply-python.com/2014/03/14/saving-output-of-nltk-text-concordance/
# concordance() function locates and prints series of phrases that contain the keyword.
# The functions only prints the output. The user is not able to save the results for further processing redirect
# the stdout

def word_phases(target_word, query_text, left_margin=10, right_margin=10):
    """
        Function to get all the phases that contain the target word in a text/passage tar_passage.

        str target_word, str tar_passage int left_margin int right_margin --> list of str
        left_margin and right_margin allocate the number of words/punctuation before and after target word
        Left margin will take note of the beginning of the text
    """
    #create list of tokens using nltk function
    tokens = nltk.word_tokenize(query_text)

    #create the text of tokens
    text = nltk.Text(tokens)

    ## Collect all the index or offset position of the target word
    c = nltk.ConcordanceIndex(query_text.tokens, key=lambda s: s.lower())

    ## Collect the range of the words that is within the target word by using text.tokens[start;end].
    ## The map function is use so that when the offset position - the target range < 0, it will be default to zero
    concordance_txt = (
    [query_text.tokens[list(map(lambda x: x - 5 if (x - left_margin) > 0 else 0, [offset]))[0]:offset + right_margin]
     for offset in c.offsets(target_word)])

    ## join the sentences for each of the target phrase and return it
    return [''.join([x + ' ' for x in con_sub]) for con_sub in concordance_txt]




def extract_phases(tokens, wordlist):
    all_phases = []
    text = nltk.Text(tokens)
    for word in wordlist:
        phases = word_phases(word, text)
        if phases:
            all_phases.append(phases)
    return all_phases


def chunk_entities(tokens):
    ## produce tags
    tagged = nltk.pos_tag(tokens)

    ## identify named entities, and don't expand NE types
    return nltk.chunk.ne_chunk(tagged, binary=True)


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


