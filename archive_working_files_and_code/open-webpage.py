# open-webpage.py

import urllib.request, urllib.error, urllib.parse, requests
from bs4 import BeautifulSoup
from bs4 import SoupStrainer


#Extract specific portions in HTML
page = requests.get('https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-728')
soup = BeautifulSoup(page.content, 'html.parser')
#interested_sect = soup.find_all('section', attrs={"aria-labelledby":"Abs1"}) #selects only the Abstract
#this can also work with this code too: interested_sect = soup.find_all('section', attrs={"data-title":"Abstract"})
print('this is the raw content: \n')
#print(str(interested_sect))

#tags = soup.find_all(['hr', 'strong'])
interested_sect1 = soup.find_all(True, {'class': ['c-article-section', 'c-article__pill-button']})
#todo this works, but Bib1 is also a c-article-section class. So I need to be working with ids
#todo id
#print(str(interested_sect1))
#soup.findAll(True, {'class':['class1', 'class2']})

interested_sect2 = soup.find_all('section', attrs=[{"data-title":"Background", "data-title":"Abstract"}])
interested_sect3 = soup.find_all('section', attrs={'id':'Bib1'})
print(str(interested_sect3))
print(str(interested_sect2))
interested_sect4 = soup.find_all(True, {'id': ['Abs1', 'Bib1']})
print(str(interested_sect4))

hello = soup.find_all(id="Bib1")

#todo <section data-title="Results"> from one article
#todo <div id="__sec5" class="tsec sec" style="user-select: auto;"> from different paper so maybe this is not possible to fix
#todo different papers have different HTML structure, so for the code to work, need to be choosing sections differently


print(str(hello))

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
    print(soup.get_text())


#with BeautifulSoup to get the text between your tags:

from bs4 import BeautifulSoup
soup = BeautifulSoup('paper.html')
print (soup.text)

#And for get the text from a specific tag just use soup.find_all :

soup = BeautifulSoup('paper.html') #To get rid of this warning, pass the additional argument 'features="lxml"' to the BeautifulSoup constructor.
for line in soup.find_all('div',attrs={"class" : "title"}):
    print (line.text)

#Extract specific portions in HTML
page = requests.get('https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-728')
soup = BeautifulSoup(page.content, 'html.parser')
interested_sect = soup.find_all('section', attrs={"aria-labelledby":"Abs1"}) #selects only the Abstract
#this can also work with this code too: interested_sect = soup.find_all('section', attrs={"data-title":"Abstract"})
print('this is the raw content: \n')
print(str(interested_sect))

#How can I select various sections and put them all in one variable? Concatenate? append? into interested_sect variable?




'''
1. Want to parse between main - content and bib1 - section:
2. if not a set function as is, then could I use regex to state start reading after you read string Abstract and
stop when you see word reference?
3. I can parse the whole thing and then delete the sections
if not needed:
4. for better performance I make variables dont_parse = regex if id=[Bib1-section] tag['id'] = 'verybold'

'''