from hdp_py import HDP, get_data
import numpy as np
import pandas as pd
import numpy.lib.scimath as sci
#Nematode

X_200, j_200 = get_nematode(max_docs=200, min_word_count=10)
cutoff = np.where(j_200==100)[0][0]
X_train, j_train, X_test, j_test = np.array(X_200)[:cutoff, :], j_200[:cutoff], np.array(X_200)[cutoff:,:], j_200[cutoff:]
vocab_size = X_train.shape[1]


%time test100 = HDP(f='categorical_fast', hypers=(50, 0.5*np.ones(50))).gibbs_direct(np.array(X_train), j_train, iters=2000, Kmax=50)

burn_in = 500

clusters_per_sim = np.zeros(2000)
for i in range(2000):
    clusters_per_sim[i] = len(np.unique(test100.direct_samples[i,:]))

fig, axn = plt.subplots(1, 2, figsize=(20,7))
images = []
images.append(axn[0].plot(np.arange(2000), clusters_per_sim))
images.append(axn[1].hist(clusters_per_sim[burn_in:]))

def topic_given_document(doc, Kmax=50, prior=0.5, burn_in=500):
    subsets = test100.direct_samples[burn_in:,j_train==doc].flatten()
    posterior = prior*np.ones(Kmax)
    for i in range(Kmax):
        posterior[i] = posterior[i] + len(subsets[subsets==i])
    return(posterior/posterior.sum())

p_t_given_d = np.apply_along_axis(topic_given_document, 1, np.unique(j_train).reshape(-1,1))

def word_given_topic(topic, vocab=vocab_size, prior=0.5, burn_in=500):
    postburn = test100.direct_samples[burn_in:,]
    subsets = postburn[np.where(postburn==topic)]
    posterior = prior*np.ones(vocab)
    for i in range(vocab):
        posterior[i] = posterior[i] + len(subsets[subsets==i])
    return(posterior/posterior.sum())

p_w_given_t = np.apply_along_axis(word_given_topic, 1, np.arange(50).reshape(-1,1))

p_w_given_d = p_w_given_t.T @ p_t_given_d.T

p_w_train = (X_train @ p_w_given_d).sum(axis=1)

perplexity_train = np.exp(-1/(p_w_train.shape[0])*(sci.log(p_w_train)).sum())


p_w_test = (X_test @ p_w_given_d).sum(axis=1)
perplexity_test = np.exp(-1/(p_w_test.shape[0])*(sci.log(p_w_test)).sum())

#Reuters

Xreuters_200, jreuters_200 = get_reuters(max_docs=200, min_word_count=10)
cutoff = np.where(jreuters_200==100)[0][0]
Xr_train, jr_train, Xr_test, jr_test = np.array(Xreuters_200)[:cutoff, :], jreuters_200[:cutoff], np.array(Xreuters_200)[cutoff:,:], jreuters_200[cutoff:]
vocab_size = Xr_train.shape[1]

%time testr100 = HDP(f='categorical_fast', hypers=(50, 0.5*np.ones(50))).gibbs_direct(np.array(Xr_train), jr_train, iters=2000, Kmax=50)


clusters_per_sim = np.zeros(2000)
for i in range(2000):
    clusters_per_sim[i] = len(np.unique(testr100.direct_samples[i,:]))

fig, axn = plt.subplots(1, 2, figsize=(20,7))
images = []
images.append(axn[0].plot(np.arange(2000), clusters_per_sim))
images.append(axn[1].hist(clusters_per_sim[burn_in:]))

burn_in = 500

def topic_given_document(doc, Kmax=50, prior=0.5, burn_in=500):
    subsets = testr100.direct_samples[burn_in:,jr_train==doc].flatten()
    posterior = prior*np.ones(Kmax)
    for i in range(Kmax):
        posterior[i] = posterior[i] + len(subsets[subsets==i])
    return(posterior/posterior.sum())

p_t_given_d = np.apply_along_axis(topic_given_document, 1, np.unique(jr_train).reshape(-1,1))

def word_given_topic(topic, vocab=vocab_size, prior=0.5, burn_in=500):
    postburn = testr100.direct_samples[burn_in:,]
    subsets = postburn[np.where(postburn==topic)]
    posterior = prior*np.ones(vocab)
    for i in range(vocab):
        posterior[i] = posterior[i] + len(subsets[subsets==i])
    return(posterior/posterior.sum())

p_w_given_t = np.apply_along_axis(word_given_topic, 1, np.arange(50).reshape(-1,1))

p_w_given_d = p_w_given_t.T @ p_t_given_d.T

p_w_train = (Xr_train @ p_w_given_d).sum(axis=1)

perplexity_train = np.exp(-1/(p_w_train.shape[0])*(sci.log(p_w_train)).sum())


p_w_test = (Xr_test @ p_w_given_d).sum(axis=1)
perplexity_test = np.exp(-1/(p_w_test.shape[0])*(sci.log(p_w_test)).sum())
