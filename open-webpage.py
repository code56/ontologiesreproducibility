# open-webpage.py

import urllib.request, urllib.error, urllib.parse

url = 'https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-728'
'''
response = urllib.request.urlopen(url)
webContent = response.read()

#print(webContent)
#save the HTML file locally

f = open('paper.html', 'wb')
f.write(webContent)
f.close


#Python - How to read HTML line by line into a list
with open("paper.html") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
#print(content) #works

#want to get the text from the HTML
with open('paper.html') as f:
    log = f.readlines()

new_log = ''
for line in log:
	new_log += line
#print("this is new log")
#print(new_log) #-this works

#create an empty dictionary to fit the key-values from the Plant Ontology database text file
#downloaded from: ....???
onto = {}

for line in open('plant-ontology-dev.txt'):
	split = line.split("\t")

#create a dictionary with keys the name of the plant ontology and value the PO term.
	if len(split) > 2:
		onto[split[1]] = split[0]
#print(onto)

#dummy string text to test the code below
text = 'spikelet is a new ontology term anthesis'

#this kinda works, but doesn't tell us where the matches were found.
#also, it matches word 'flower' instead of how it is in the text 'flowering'
#so need to add further code to read the full word before quering the onto dictionary

for query, onto_id in onto.items():
    if query in new_log:
        print("this match was found at")
        print(query, onto_id)

#must find a way to record from where in the HTML structure the match was found.
'''

from bs4 import BeautifulSoup
from bs4 import SoupStrainer

def has_class_but_no_id(tag):
    return tag.has_attr('main-content') and not tag.has_attr('Bib1-section')

with open('paper.html') as f:
    #soup = BeautifulSoup(f, 'html.parser')
    #print(soup.prettify())
    # Will parse only the below mentioned "ids".
    parse_only = SoupStrainer(id=["main-content"]) #but want to exclude Bib1-section (bibliography)
    soup = BeautifulSoup(f, "html.parser", parse_only=parse_only)
    hello = soup.find_all(has_class_but_no_id)
    print(hello)
    #print(soup.get_text())


1. Want to parse between main - content and bib1 - section:
2. if not a set function as is, then could I use regex to state start reading after you read string Abstract and
stop when you see word reference?
3. I can parse the whole thing and then delete the sections
if not needed:
4. for better performance I make variables dont_parse = regex if id=[Bib1-section] tag['id'] = 'verybold'
