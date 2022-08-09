#!/usr/bin/python

import pandas as pd
import os, os.path, pdftotext, re, string, nltk, argparse, sys, functools, operator
import glob, os
from pathlib import Path


# Pre-requisite: put the pdf version of the articles you want to process in the papers_collection_folder

yourpath = os.getcwd()
data_folder = Path("papers_collection_folder")
text_article_folder = Path("text_article_folder")

folder = yourpath / Path(data_folder)
files_in_folder = data_folder.iterdir()
for item in files_in_folder:
    if item.is_file():
        print(item.name)
        pdfarticlename = item.name
    exec(open("Deus_ex_machina.py").read())


#TODO feed item.name to Deus ex machina as a parameter






