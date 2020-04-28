import urllib
import string
from itertools import compress
from nltk.corpus import stopwords 
import pandas as pd
from functools import reduce
import numpy as np


def docsToList(data):
    '''
    This function takes a string of abstracts and converts it to a list of lists of the words in each abstract.
    This function was made specifically for the data obtained here:
    https://raw.githubusercontent.com/tdhopper/topic-modeling-datasets/master/data/raw/Nematode%20biology%20abstracts/cgcbib.txt
    '''
    
    # Remove '\n' and '\r'
    data = data.lower().translate(str.maketrans('\n', ' '))
    data = data.translate(str.maketrans('\r', ' '))
    
    # Remove punctuation except for '-' so we can split after each abstract
    data = data.translate(str.maketrans('', '', '!"#$%&\'()*+,./;<=>?@[\\]^_`{|}~'))
    
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
    data = [abstract.split('abstract:')[1] for abstract in data]
    
    # Remove any remaining :'s
    data = [abstract.translate(str.maketrans(':', ' ')) for abstract in data]
    
    # Remove abstracts that only state 'in french'
    not_french = ['in french' not in i for i in data]
    data = list(compress(data, not_french))
    
    # Create list of lists output
    output = [i.split() for i in data]
    
    return output


def reducedVocab(lists, stop_words = None, min_word_count = 10):
    '''
    This function takes a list of words in a list of documents and returns the lists of lists with a reduced
    vocabulary, the flattened list, and the vocabulary
    '''
    
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


def listsToVec(lists, stop_words = None, min_word_count = 10, verbose = 1):
    '''
    This function takes a list of lists of the words in each document. It removes any stop words, removes words that
    appear 'min_word_count' times or less, and maps each word in the documents' vocabulary to a number. 
    Returns: data matrix X, where each row is a draw from a categorical distribution representing one word
             vector j encoding the corresponding documents each word belongs to'''

    # Remove stop words and words that appear less than 'min_word_count' times
    docs, one_list, vocab = reducedVocab(lists, stop_words, min_word_count)
    
    # Map each word to a number
    #numbers = list(range(len(vocab)))
    #vocab_dict = dict(zip(vocab, numbers))
    #x = list(map(vocab_dict.get, one_list))
    
    # Check for empty lists and print warning if one is found
    counter = 0
    for i in range(len(docs)-1 ,-1, -1):
        if len(docs[i]) == 0:
            if verbose > 1:
                print(f'WARNING: Document {i} is empty and being removed...')
            del docs[i]
            counter += 1
    
    if verbose == 1 and counter > 1:
        print(f'WARNING: {counter} documents are empty and being removed...')
    
    elif verbose == 1 and counter == 1:
        print(f'WARNING: {counter} document is empty and being removed...')
    
    X_matrix = pd.DataFrame(np.zeros((len(one_list), len(vocab))),
                           columns=vocab)

    for i, word in enumerate(one_list):
        X_matrix.loc[i, word] = 1   
    
    # Determine which document each word belongs to
    count, j = 0, []
    for i in docs:
        j.append([count]*len(i))
        count += 1
        
    # Reduce to a flattened list
    j = [i for sublist in j for i in sublist]
    
    return X_matrix.astype('int'), np.array(j)


### DATA GETTING FUNCTIONS


def get_nematode(max_docs = None, min_word_count = 1):
    """
    Returns the data matrix X and document encodings j from the nematode abstracts
    used in the HDP paper.
    """

    url = 'https://raw.githubusercontent.com/tdhopper/topic-modeling-datasets/master/data/raw/Nematode%20biology%20abstracts/cgcbib.txt'
    file = urllib.request.urlopen(url)
    data = file.read().decode("ISO-8859-1")
    
    lists = docsToList(data)
    if max_docs is None:
        max_docs = len(lists)
    return listsToVec(lists[:max_docs], min_word_count=min_word_count)


def get_reuters(max_docs = None, min_word_count = 1, data_dir = '../data'):
    """
    Returns a list of list of words in the Reuters data.
    data_dir: a path to the directory containing the pre-downloaded Reuters data.
    """
    
    directory = os.fsencode('../data')
    docs = []
    for file in os.listdir(directory):
        root = directory.decode('ascii')
        filename = os.fsdecode(file)
        f = open(f'{root}/{filename}', 'r')
        data= f.read()
        soup = BeautifulSoup(data)
        contents = soup.findAll('text')
        f.close()
        docs.append(str(contents).split('</text>'))

    docs = [i for doc in docs for i in doc]
    # split on </dateline> and keep everything after it
    docs = list(compress(docs, ['</dateline>' in i for i in docs]))
    docs = [i.split('</dateline>')[1] for i in docs]
    docs = [i.lower().translate(str.maketrans('\n', ' ')) for i in docs]
    docs = [i.translate(str.maketrans('\r', ' ')) for i in docs]
    docs = [i.translate(str.maketrans('\x03', ' ')) for i in docs]
    docs = [i.translate(str.maketrans('', '', string.punctuation)) for i in docs]
    docs = [i.translate(str.maketrans('', '', string.digits)) for i in docs]
    docs = [i.replace('said',' ') for i in docs] # another stop word
    docs = [i.replace('reuter', ' ') for i in docs] # the name of the company at the end of most articles
    docs = [i.split() for i in docs]
    
    return docs


    
