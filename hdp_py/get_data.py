import urllib
import string
from itertools import compress
from nltk.corpus import stopwords 
import pandas as pd
from functools import reduce
import numpy as np
import re
import os
import pkgutil
from bs4 import BeautifulSoup
import gensim
import gensim.corpora as corpora


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


def get_nematode(max_docs = None, min_word_count = 1, LDA = False):
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
    
    if LDA = False:
        return listsToVec(lists[:max_docs], min_word_count=min_word_count)
    else:
        return reducedVocab(lists[:max_docs], min_word_count = min_word_count)

def get_reuters(max_docs = None, min_word_count = 1, LDA = False):
    """
    Returns the data matrix X and document encodings j in the Reuters data.
    data_dir: a path to the directory containing the pre-downloaded Reuters data.
    """
    
    #directory = pkgutil.get_data('hdp_py', 'data') #os.fsencode(data_dir)
    docs = []
    for i in range(22):
        suffix = '%03i' % i
        #root = directory.decode('ascii')
        #filename = os.fsdecode(file)
        #print(filename)
        #f = open(filename, 'r')
        data = pkgutil.get_data('hdp_py', f'data/reut2-{suffix}.sgm')
        soup = BeautifulSoup(data, features='lxml')
        contents = soup.findAll('text')
        #f.close()
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
    
    if max_docs is None:
        max_docs = len(docs)
    
    if LDA = False:
        return listsToVec(docs[:max_docs], min_word_count=min_word_count)
    else:
        return reducedVocab(docs[:max_docs], min_word_count = min_word_count)


def get_test_data(N, L, Jmax):
    """
    Returns the data matrix X and group encodings j for a random set of multinomial data.
    X is an (N,L) matrix and j is an (N,) vector with values drawn from [0,Jmax-1]
    """
    
    j = np.random.choice(Jmax, size=N)
    Xtest = np.zeros((N, L), dtype='int')
    col_choices = np.random.choice(L, size=N)
    Xtest[range(N), col_choices] = 1
    return Xtest, j


def get_simulated_pop_data():
    """
    Returns the data matrix X, study encodings j, and latent study-group information
    z for 3 simulated studies of ant populations. Each row corresponds to a unique trial
    """
    np.random.seed(111)
    Study1_rates = np.random.uniform(low=0, high=50, size=4)
    np.random.seed(112)
    Study1_rates[3] = Study1_rates[2] + Study1_rates[1] + np.random.uniform(low=-.1,high=.1)*Study1_rates[2]*Study1_rates[1]
    np.random.seed(222)
    Study2_rates = np.array((Study1_rates[0]+np.random.uniform(low=-0.5, high=0.5), 
                             np.random.uniform(low=0, high=50), 
                             Study1_rates[2]+np.random.uniform(low=-0.5, high=0.5), 
                             np.random.uniform(low=0, high=50)))
    np.random.seed(223)
    Study2_rates[3] = Study2_rates[2] + Study2_rates[1] + np.random.uniform(low=-.1,high=.1)*Study2_rates[2]*Study2_rates[1]
    np.random.seed(333)
    Study3_rates = np.random.uniform(low=0, high=50, size=4)
    np.random.seed(334)
    Study3_rates[0] = Study2_rates[0]+np.random.uniform(low=-0.5, high=0.5)
    np.random.seed(335)
    Study3_rates[3] = Study3_rates[2] + Study3_rates[1] + np.random.uniform(low=-.1,high=.1)*Study3_rates[2]*Study3_rates[1]
    
    
    #Each set of conditions in study 1 done 20 times, study 2 16 times,
    #study 3 10 times:
    np.random.seed(113)
    study1_obs = np.random.poisson(lam=Study1_rates, size=(20,4))
    np.random.seed(224)
    study2_obs = np.random.poisson(lam=Study2_rates, size=(16,4))
    np.random.seed(336)
    study3_obs = np.random.poisson(lam=Study3_rates, size=(10,4))
    
    pop_obs = np.concatenate((study1_obs.flatten(), study2_obs.flatten(), study3_obs.flatten()))
    study_tracker = np.repeat(np.array(["S1", "S2", "S3"]), [20*4, 16*4, 10*4])
    cond_tracker = np.concatenate(np.array((["Control", "Alt", "Temp", "Alt + Temp"]*20, 
                                            ["Control", "Light", "Temp", "Light + Temp"]*16, 
                                            ["Control", "Food", "Dirt", "Food + Dirt"]*10)).flatten())
    study_factor = np.unique(study_tracker, return_inverse=True)[1]
    return pop_obs[:, None], study_factor, cond_tracker
    
    
 ### LDA FUNCTIONS

def LDA_preprocessing(data, n_documents, test_size, min_word_count):
    '''
    This function takes in data formatted by any of the get_{topic}_data functions with LDA = true called.
    
    n_documents: the number of documents to select from data. 
    test_size: the proportion of n_documents that should be held out for testing
    min_word_count: the minimum number of times a word should appear to be kept in vocabulary
    
    This function returns id2word and corpus for LDA training and testing
    '''
    
    selected = np.random.choice(len(data), n_documents, replace = False)
    subset_data = [data[i] for i in selected]
    
    docs, one_list, vocab = reducedVocab(subset_data, min_word_count = min_word_count)
    
    cut_off = int(np.floor(n_documents * test_size))
    train, test = docs[:cut_off], docs[cut_off:]
    
    id2word = corpora.Dictionary(docs)
    train_corpus = [id2word.doc2bow(doc) for doc in train]
    test_corpus = [id2word.doc2bow(doc) for doc in test]
    
    return id2word, train_corpus, test_corpus, test

def LDA(id2word, corpus, n_topics):
    '''
    This function runs gensim's LdaModel.
    '''
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                               id2word = id2word,
                                               num_topics = n_topics,
                                               random_state = 23,
                                               alpha = 'asymmetric',
                                               iterations = 500)
    
    return lda_model

def perplexity(model, test_corpus, test):
    '''
    This function takes a trained LDA model and calculates the perplexity of the test corpus
    '''
    
    model.get_document_topics(test_corpus, minimum_probability = 1e-8, per_word_topics = True)       
    
    new_topics = model[test_corpus]
    
    log_perplex = 0

    for i in range(len(test_corpus)):
        theta = [e for _, e in new_topics[i][0]]
        phi = []
        for j in range(len(new_topics[i][2])):
            first, second = new_topics[i][2][j]
            for k in range(len(theta)):
                phi.append([e for _, e in second if _ == k])
                if len(phi[j*len(theta) + k]) == 0:
                    phi[j*len(theta) + k] = [0]
        phi = np.array(phi).reshape(-1, len(theta))
        log_perplex -= np.sum(np.log(np.inner(theta, phi)))

    N = len([i for sublist in test for i in sublist])

    return np.exp(log_perplex / N)

def plt_perplexity(perplexity, min_topics, max_topics):
    '''
    This function plots the perplexity given perplexity array.
    
    First row of perplexity array is the perplexity values
    Second row of perplexity array is the corresponding number of topics used for LDA training
    '''
    plt.plot(perplexity[1,:], perplexity[0,:])
    plt.xlabel('Number of LDA Topics')
    plt.ylabel('Perplexity')
    plt.title('Perplexity of LDA Model on Test Documents')
    plt.show()
