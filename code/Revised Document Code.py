# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:58:45 2020

@author: morri
"""


import urllib
import string
from itertools import compress
from nltk.corpus import stopwords 
import pandas as pd
from functools import reduce
import re
import json

url = 'https://raw.githubusercontent.com/tdhopper/topic-modeling-datasets/master/data/raw/Nematode%20biology%20abstracts/cgcbib.txt'
file = urllib.request.urlopen(url)
data = file.read().decode("ISO-8859-1")

#-------------------------------------------------------------------
#Updating docsToList function:

def docsToList(data):
    '''This function takes a string of abstracts and converts it to a list of lists of the words in each abstract.
       This function was made specifically for the data obtained here:
       https://raw.githubusercontent.com/tdhopper/topic-modeling-datasets/master/data/raw/Nematode%20biology%20abstracts/cgcbib.txt'''
    
    # Remove '\n' and '\r'
    data = data.lower().translate(str.maketrans('\n', ' '))
    data = data.translate(str.maketrans('\r', ' '))
    
    # Remove punctuation except for '-' so we can split after each abstract
    data = data.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
    
    # Remove numbers
    data = data.translate(str.maketrans('','', string.digits))
    
    # Split after 'abstract' is stated
    data = data.split('-------------------')
    # Remove '-' punctuation now
    data = [abstract.translate(str.maketrans('-', ' ')) for abstract in data]
    
    # Remove entries without the word "abstract" in it
    abs_check = ['abstract' in i for i in data]
    data = list(compress(data, abs_check))

    # Only keep the words after 'abstract'
    data = [abstract.split('abstract')[1] for abstract in data]
    
    # Remove abstracts that only state 'in french'
    not_french = ['in french' not in i for i in data]
    data = list(compress(data, not_french))
    
    # Create list of lists output
    output = [i.split() for i in data]
    
    return output


def docsToListRevised(data):
    '''This function takes a string of abstracts and converts it to a list of lists of the words in each abstract.
       This function was made specifically for the data obtained here:
       https://raw.githubusercontent.com/tdhopper/topic-modeling-datasets/master/data/raw/Nematode%20biology%20abstracts/cgcbib.txt'''
    
    # Remove '\n' and '\r'
    data = data.lower().translate(str.maketrans('\n', ' '))
    data = data.translate(str.maketrans('\r', ' '))
    data = data.translate(str.maketrans('','', '!"#$%&\'()*+,./;<=>?@[\\]^_`{|}~'))
    data = data.translate(str.maketrans('','', string.digits))
    
    # Removing excess white space
    data = ' '.join(data.split())
    
    # Setting up the dictionary form
    data = re.sub('key: ', '{"key":"', data)
    data = re.sub('medline: ', '","medline":"', data)
    data = re.sub('authors: ', '","authors":"', data)
    data = re.sub('title: ', '","title":"', data)
    data = re.sub('citation: ', '","citation":"', data)
    data = re.sub('type: ', '","type":"', data)
    data = re.sub('genes: ', '","genes":"', data)
    data = re.sub('abstract: ', '","abstract":"', data)
    data = re.sub('-------------------', '"}>>>', data)
    
    # Splitting on the delimiter created
    data = data.split('>>>')[:-1]
    
    # Using json.loads() to convert to dictionary, and only keep
    # the values associated with the 'abstract' key
    data = [json.loads(abstract)['abstract'] for abstract in data]
    
    # Remove abstracts that only state 'in french'
    not_french = ['in french' not in i for i in data]
    data = list(compress(data, not_french))
    
    # Create list of lists output
    output = [i.split() for i in data]
    
    return output

data2 = docsToList(data)
data3 = docsToListRevised(data)

len(data2[64])
len(data3[64])

#-------------------------------------------------------------------
#Updating listToVec function:
def reducedVocab(lists, stop_words = None, min_word_count = 10):
    '''This function takes a list of words in a list of documents and returns the lists of lists with a reduced
       vocabulary, the flattened list, and the vocabulary'''
    
    if stop_words == None:
        stop_words = set(stopwords.words('english'))
    
    # Remove stop words
    words = [i for sublist in lists for i in sublist if not i in stop_words]

    # Remove words that appear less than min_word_count times
    wordSeries = pd.Series(words)
    vocab = list(compress(wordSeries.value_counts().index, wordSeries.value_counts() >= min_word_count))
    
    # Recreate lists with filtered vocab
    docs = []
    for j in range(len(lists)):
        docs.append([i for i in lists[j] if i in vocab])
    
    #flatten docs
    one_list = [i for sublist in docs for i in sublist]
    
    return docs, one_list, vocab

def listsToVecRevised(lists, stop_words = None, min_word_count = 10, verbose = 1):
    '''This function takes a list of lists of the words in each document. It removes any stop words, removes words that
       appear 10 times or less, and maps each word in the documents' vocabulary to a number. Two flattened vectors are
       returned, the mapped numbers 'x', and the corresponding document each word belongs to 'j'.'''

    # Remove stop words and words that appear less than 'min_word_count' times
    docs, one_list, vocab = reducedVocab(lists, stop_words, min_word_count)
    
    # Map each word to a number
    numbers = list(range(len(vocab)))
    vocab_dict = dict(zip(vocab, numbers))
    x = list(map(vocab_dict.get, one_list))
    
    # Check for empty lists and print warning if one is found
    '''counter = 0
    for i in range(len(docs)-1 ,-1, -1):
        if len(docs[i]) == 0:
            if verbose > 1:
                print(f'WARNING: Document {i} is empty and being removed...')
            del docs[i]
            counter += 1
    
    if verbose == 1 and counter > 1:
        print(f'WARNING: {counter} documents are empty and being removed...')
    
    elif verbose == 1 and counter == 1:
        print(f'WARNING: {counter} document is empty and being removed...')'''
    good_indices = [i > 0 for i in list(map(len, docs))]
    docs2 = list(compress(docs, good_indices))
    counter = len(docs) - len(docs2)
    if verbose == 1 and counter > 1:
        print(f'WARNING: {counter} documents are empty and being removed...')
    
    elif verbose == 1 and counter == 1:
        print(f'WARNING: {counter} document is empty and being removed...')
    
    # Determine which document each word belongs to
    count, j = 0, []
    for i in docs2:
        j.append([count]*len(i))
        count += 1
        
    # Reduce to a flattened list
    j = [i for sublist in j for i in sublist]
    
    return x,j

