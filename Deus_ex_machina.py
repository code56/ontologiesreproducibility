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
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
import urllib.request, urllib.error, urllib.parse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import requests, pprint
import bs4
from bs4 import BeautifulSoup
from itertools import repeat
import csv



class File(object):

    def __init__(self, name):
        self.name = name

    def convert_pdf_to_text(self, name):
        #with open(name, 'rb') as f:

        with open("papers_collection_folder/" + name, 'rb') as f:
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
           # first_line = f.readline()
           # print('first line', first_line)

            log = f.readlines()
        text_article = ''
        for line in log:
            text_article += line
        return text_article



    #publication date, keywords: Published:, Accepted:, first sentence year date, avoid references, Received:

    # pre-processing of text_article. Remove punctuation
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
        res = [' '.join(tups) for tups in
               bigram]  # ['. BMC', 'BMC Genomics', 'Genomics 2013', 'RESEARCH ARTICLE', 'ARTICLE Open']
        bigram_matches = []
        for query, onto_id in dict_po.items():
            if query in res:
                print('found query match with bigram match', " | " + onto_id + " | " + query)
                bigram_matches.append(query + " | " + onto_id)
        print(bigram_matches)
        return bigram_matches

    def find_ngrams(self, data_tokenised, dict_po):
        ngram_matches = []
        n = 6
        for i in range(1, n + 1):
            n_grams = list(ngrams(data_tokenised, i))
            res = [' '.join(tups) for tups in n_grams]
            for query, onto_id in dict_po.items():
                if query in res:
                    ngram_matches.append((query,onto_id))
        return ngram_matches

    def get_all_phrases_containing_tar_wrd(self, target_word, tar_passage, left_margin=5, right_margin=5):
        # code for this function adjusted from https://simply-python.com/2014/03/14/saving-output-of-nltk-text-concordance/
        ## Create list of tokens using nltk function
        tokens = nltk.word_tokenize(tar_passage)

        ## Create the text of tokens
        text = nltk.Text(tokens)

        ## Collect all the index or offset position of the target word
        c = nltk.ConcordanceIndex(text.tokens, key=lambda s: s.lower())

        ## Collect the range of the words that is within the target word by using text.tokens[start;end].
        ## The map function is use so that when the offset position - the target range < 0, it will be default to zero
        concordance_txt = (
        [text.tokens[list(map(lambda x: x - 5 if (x - left_margin) > 0 else 0, [offset]))[0]:offset + right_margin] for
         offset in c.offsets(target_word)])

        ## join the sentences for each of the target phrase and return it
        return [''.join([x + ' ' for x in con_sub]) for con_sub in concordance_txt]

    def processing_array_express_info(self, article_text):
        accession_url = "https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/"
        accession_numbers_in_article = re.findall("E-[A-Z]{4}-[0-9]*", article_text)
        set_article_accession_numbers = set(accession_numbers_in_article)  # {'E-MTAB-1729'}
        if len(set_article_accession_numbers) == 0:
            return None
        else:
            for accession_number in set_article_accession_numbers:
                api_url_concatenated = accession_url + str(accession_number)
                getxml = requests.request('GET', api_url_concatenated)
                file = open('%s.txt' % accession_number, 'w')
                file.writelines(getxml.text)
                file.close()
                soup = bs4.BeautifulSoup(getxml.text, 'xml')
                metadata = []
                for hit in soup.find_all("value"):
                    metadata.append(hit.text.strip())
            return set_article_accession_numbers, metadata


    #to bypass issue with EBI blocking API-Request if too many hits in a short period of time for the same search
    #from the same IP address
    def processing_xml(self, article_text):
        accession_url = "https://www.ebi.ac.uk/arrayexpress/xml/v3/experiments/"
        accession_numbers_in_article = re.findall("E-[A-Z]{4}-[0-9]*", article_text)
        set_article_accession_numbers = set(accession_numbers_in_article)  # {'E-MTAB-1729'}
        if len(set_article_accession_numbers) == 0:
            return None
        else:
            for accession_number in set_article_accession_numbers:
                api_url_concatenated = accession_url + str(accession_number)
                page = ''
                while page == '':
                    try:
                        page = requests.get(api_url_concatenated)
                        file = open('%s.txt' % accession_number, 'w')
                        file.writelines(page.text)
                        file.close()
                        soup = bs4.BeautifulSoup(page.text, 'xml')
                        metadata = []
                        for hit in soup.find_all("value"):
                            metadata.append(hit.text.strip())


                    except:
                        print("Connection refused by the server...will retry in 5 seconds")
                        time.sleep(5)
                        print("5 seconds have passed, now let me continue...")
                        continue

                        #getxml = requests.request('GET', api_url_concatenated)
                        file = open('%s.txt' % accession_number, 'w')
                        file.writelines(page.text)
                        file.close()
                        soup = bs4.BeautifulSoup(page.text, 'xml')
                        metadata = []
                        for hit in soup.find_all("value"):
                            metadata.append(hit.text.strip())
                return set_article_accession_numbers, metadata

    def find_project_accession_number(self, article_text, accession_url):
        project_accession_numbers_in_article = re.findall("PRJ[E|D|N][A-Z][0-9]+", article_text)
        set_project_accession_numbers = set(project_accession_numbers_in_article)  # {'PRJEB12345'}
        if len(set_project_accession_numbers) == 0:
            return None
        else:
            for project_accession_number in set_project_accession_numbers:
                api_url_concatenated = accession_url + str(project_accession_number)
                getxml = requests.request('GET', api_url_concatenated)
                file = open('%s.txt' % project_accession_number, 'w')
                file.writelines(getxml.text)
                file.close()
            return project_accession_number



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

header = ['article file name', 'article ontologies', 'ArrayExpress number', 'project accession number', 'XML metadata',
          'XML ontologies', 'common ontologies', 'article ontologies score', 'ArrayExpress score','XML file ontologies score', 'Project Accession score','common ontologies score','Reproducibility Metric Score (RMS)']

with open('scores.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)


def main(articlename):
    file1 = File(articlename)

    convertpdf = file1.convert_pdf_to_text(file1.name)
    textarticlecreation = file1.create_text_file(convertpdf)
    text_article = file1.create_text_article(convertpdf)

    ngram_matches = file1.find_ngrams(textarticlecreation, po_dict)
    ngram_matches_score = 0
    if ngram_matches is not None:
        ngram_matches_score = ngram_matches_score + 1




    #from sklearn.feature_extraction.text import CountVectorizer
    #from nltk.tokenize import RegexpTokenizer

    # tokenizer to remove unwanted elements from out data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z]+')

    data_reproducibility_keywords = ['accession', 'accessions', 'data', 'Supporting', 'available', 'repository', 'GO',
                                     'EBI',
                                     'ArrayExpress', 'PO', 'sequences', 'expression', 'snps', 'genes', 'wheat', 'rice']
    phrases_from_article = []
    for word in data_reproducibility_keywords:
        phrases_from_article = file1.get_all_phrases_containing_tar_wrd(word, text_article)
       # phrases_from_article.append(file1.get_all_phrases_containing_tar_wrd(word, text_article))
       # print('phrases in article', phrases_from_article)
       #print('phrases from text article:', word, phrases_from_article)

       # it doesn't return all the occurences or the complete sentences because it is two column text.
       #TODO produce a list instead of printing them individually & include in the scores.csv file


    #xml_metadata = file1.processing_array_express_info(text_article)
    xml_metadata = file1.processing_xml(text_article)

    score_for_xml_ontology_matching = 0
    article_accessionnumber_score = 0
    xml_and_onto_matching =[]
    if xml_metadata is not None:
        if len(xml_metadata) == 0:
            print('xml metadata is empty')
        else:
            array_express_number = xml_metadata[0]
            arrayexpress_score = article_accessionnumber_score + 1
            for query, onto_id in po_dict.items():
                if query in xml_metadata[1]:
                    xml_and_onto_matching.append((query, onto_id))
                    score_xml = score_for_xml_ontology_matching + 1
    else:
        score_xml = score_for_xml_ontology_matching
        xml_and_onto_matching = None
        array_express_number = None
        arrayexpress_score = 0

    # other accession numbers. e.g. GenBank HP608076 - HP639668 . See accession number prefixes: https://www.ncbi.nlm.nih.gov/genbank/acc_prefix/
    # so can do another Regex, to find other accession. It is a long list (as per the link above) and I am not sure which one of those are
    # relevant to crop transcriptomics.

    # for studies such as ontopaper2 which dont have Array express accession numbers, but maybe including SRP numbers
    # https://www.ebi.ac.uk/ena/browser/view/PRJDB2496?show=reads
    # https://www.ebi.ac.uk/ena/browser/api/xml/DRP000768?download=true

    # Do similar thing for finding ENA project reference number.
    # https://www.ebi.ac.uk/ena/browser/view/PRJDB2496?show=reads
    # guide from ENA: https://ena-docs.readthedocs.io/en/latest/submit/general-guide/accessions.html
    # Projects: PRJ(E|D|N)[A-Z][0-9]+ e.g. PRJEB12345.
    # Studies: (E|D|S)RP[0-9]{6,} e.g.ERP123456


    project_url_to_concatenate = 'https://www.ebi.ac.uk/ena/browser/api/xml/'
    project_accession = file1.find_project_accession_number(text_article, project_url_to_concatenate)

    # computing score for assessment of finding Project Accession number or not. If found add 2 to the score
    score_for_project_accession_number = 0
    if project_accession is not None:
        score_for_finding_project_accession_number = score_for_project_accession_number + 2
    else:
        score_for_finding_project_accession_number = score_for_project_accession_number



    #common ontologies comparison between the ontologies found in research article and XML metadata
    common_ontologies = []

    score_for_finding_common_ontologies_between_journal_xml = 0
    var1 = 0
    if ngram_matches and xml_and_onto_matching is not None:
        for tuple_element in ngram_matches:
            if tuple_element in xml_and_onto_matching:
                common_ontologies.append(tuple_element)
                var1 = score_for_finding_common_ontologies_between_journal_xml + 2

    #write the outputs and scores in the score.csv file
    data = [[file1.name, ngram_matches, array_express_number, project_accession, xml_metadata, xml_and_onto_matching, common_ontologies, ngram_matches_score, arrayexpress_score, score_xml, score_for_finding_project_accession_number, var1, (ngram_matches_score + arrayexpress_score + score_xml + score_for_finding_project_accession_number + var1)]]

    with open('scores.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


if __name__ == "__main__":

    for file in os.listdir("papers_collection_folder"):
        if file.endswith(".pdf"):
            print(file)
            main(file)


#TODO for more tidy code, have the code create the text files and put them in a different directory, not the main directory.
#TODO code for if it has more than one project accession & when there is more than one ArrayExpress found?
#TODO read the phrases, 'accession' and add a score. Or make some more Regex searches
#TODO parse article text ignoring references (References, References and Notes, LITERATURE CITED, REFERENCES, REFERENCES AND NOTES
#TODO have the code create the article text files neatly at a separate folder, the same for project accession xml files
#TODO add more ontology vocabularies (e.g. crop ontologies)