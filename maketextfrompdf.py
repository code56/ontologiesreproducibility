#this is a python script to take a pdf (use case: wheat transcriptomics article) and convert it to text
#stylistic information does not matter
#important information to be sourced from the conversion is methods and other key words & phrases
#next step make this work for more than one papers. Read & work on papers in a folder automatically 
#also, need to have the CO and PO to query the document by. (at the moment is matching against PO db)

import pandas as pd
import os, os.path, sys, pdftotext, nltk, re, string


from pathlib import Path

yourpath = os.getcwd()

data_folder = Path("papers_collection_folder") #can be changed to user input data
folder = yourpath / Path(data_folder)
print("this is directory_to_folder", folder)
files_in_folder = data_folder.iterdir()
for item in files_in_folder:
    if item.is_file():
        print(item.name)


#
# def find_the_dir(yourpath, directory_to_folder):
#     for root, dirs, files in os.walk(yourpath, topdown=False):
#         for name in files:
#             print("this is name in files", os.path.join(root, name))
#             return
#
#         for name in dirs:
#             if os.path.join(root, name) == str(directory_to_folder):
#                 print("Found the subdirectory with the research papers!")
#                 # put the pdf files in this subdirectory in a list.
#
#             else:
#                 pass
#
# find_the_dir(yourpath, directory_to_folder)

    #for name in dirs:
     #   dirs1 = os.path.join(root, name)
      #  print("found a subdirectory", dirs1)
       # if dirs1 == str(directory_to_folder):
        #    print("found the data_folder")
        #stuff


#get all pdf files from the directory
#for filename in os.listdir(os.getcwd()):
 #  with open(os.path.join(os.cwd(), filename), 'r') as f: # open in readonly mode
      # do your stuff
  #      print("I found the papers")


# Load your PDF
with open("ontopaper_usecase.pdf", "rb") as f:
    pdf = pdftotext.PDF(f)

# If it's password-protected
with open("ontopaper_usecase.pdf", "rb") as f:
    pdf = pdftotext.PDF(f, "secret")

# How many pages?
print(len(pdf))

# Iterate over all the pages
# Write the pdf contents in a file called output.txt
print("Writing into an output.text file, hold on")

for page in pdf:
    #print(page)
    f = open("output.txt", "w+")
    f.write("\n\n".join(pdf))
    f.close()
print("done writing to output.text file")


with open('output.txt') as f:
	log = f.readlines()

text_article = ''
for line in log:
	text_article += line

#print('this is the text version of the pdf journal article after being converted with pdftotext to text', text_article) #the text version of the pdf article

#create an empty dictionary to fit the key-values from the Plant Ontology database text file 
#downloaded from: https://raw.githubusercontent.com/Planteome/plant-ontology/master/plant-ontology.txt

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

tokenised_data = nltk.word_tokenize(text_article)


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



#dummy string text to test the code below
#text = 'spikelet is a new ontology term anthesis'

#but this is wrong as it will confuse 'stem' with 'system'.
for query, onto_id in onto.items():
	if query in text_article:
		print(query, onto_id)  #print the key found in the string of text and its pair from the dictionary 

#downloads the HTML of a website. For full-text
import urllib.request

with urllib.request.urlopen('https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-728') as response:
    html=response.read()
    #print(html)

#Python - How to read HTML line by line into a list
    with open(html) as f:
        content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

'''
@LinuxUser so the problem is how to get the html properly to parse it with BeautifulSoup? Maybe you should try to retrieve first the page
 by using requests and then process the HTML content with BeautifulSoup as shown. Also if the problem with processing the alt content is resolved 
 when reading it from a file, I would suggest to make a different question to fix that and accept/close this one, as I think the given solutions are valid

'''
from bs4 import BeautifulSoup

import requests
import re

# open-webpage.py

import urllib.request, urllib.error, urllib.parse


url = 'https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-728'

response = urllib.request.urlopen(url)
webContent = response.read()

print(webContent[0:300])

