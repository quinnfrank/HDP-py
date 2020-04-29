# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:30:08 2020

@author: morri
"""
#import HDP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Simulated example
#We'll use a Poisson version here. An example use case of the Poisson
#HDP is if there are multiple studies of counts of ant populations
#with some overlapping conditions (e.g., one study tests for 
#temperature and sunlight exposure, another study tests for temperature
#and altitude - we would hope that even under different studies,
#any control group that has the same conditions would be clustered
#together. If not, it could be indication that the studies are not
#comparable due to other variables that were not controlled.

#Population size rates in each population
Study1_rates = np.random.uniform(low=0, high=50, size=4)
Study1_rates[3] = Study1_rates[2] + Study1_rates[1] + np.random.uniform(low=-.1,high=.1)*Study1_rates[2]*Study1_rates[1]
Study2_rates = np.array((Study1_rates[0]+np.random.uniform(low=-1.5, high=1.5), 
                         np.random.uniform(low=0, high=50), 
                         Study1_rates[2]+np.random.uniform(low=-1.5, high=1.5), 
                         np.random.uniform(low=0, high=50)))
Study2_rates[3] = Study2_rates[2] + Study2_rates[1] + np.random.uniform(low=-.1,high=.1)*Study2_rates[2]*Study2_rates[1]
Study3_rates = np.random.uniform(low=0, high=50, size=4)
Study3_rates[0] = Study2_rates[0]+np.random.uniform(low=-1.5, high=1.5)
Study3_rates[3] = Study3_rates[2] + Study3_rates[1] + np.random.uniform(low=-.1,high=.1)*Study3_rates[2]*Study3_rates[1]


#Each set of conditions in study 1 done 20 times, study 2 16 times,
#study 3 10 times:
study1_obs = np.random.poisson(lam=Study1_rates, size=(20,4))
study2_obs = np.random.poisson(lam=Study2_rates, size=(16,4))
study3_obs = np.random.poisson(lam=Study3_rates, size=(10,4))

pop_obs = np.concatenate((study1_obs.flatten(), study2_obs.flatten(), study3_obs.flatten()))
study_tracker = np.repeat(np.array(["S1", "S2", "S3"]), [20*4, 16*4, 10*4])
cond_tracker = np.concatenate(np.array((["Control", "Alt", "Temp", "Alt + Temp"]*20, 
                                        ["Control", "Light", "Temp", "Light + Temp"]*16, 
                                        ["Control", "Food", "Dirt", "Food + Dirt"]*10)).flatten())
study_factor = np.unique(study_tracker, return_inverse=True)[1]
%time c = HDP(f='poisson', hypers=(6,1)).gibbs_direct(pop_obs[:,None], study_factor, iters=500, Kmax=12)
#%time d = HDP(f='poisson', hypers=(6,1)).gibbs_cfr(pop_obs[:,None], study_factor, iters=300, Kmax=12)

print(np.unique(c.direct_samples.T[:,450:500]))

clusters_per_sim = np.zeros(450)
for i in range(450):
    clusters_per_sim[i] = len(np.unique(c.direct_samples.T[:,i+50]))
print(pd.Series(clusters_per_sim).value_counts())
fig, axn = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(20,7))
images=[]
counter = 0
for i in range(3):
    for j in range(4):
        title = r'$\beta_{' + str(counter) + '}$'
        data = c.beta_samples[:,counter]
        images.append(axn[i,j].plot(np.arange(500), data))
        axn[i, j].label_outer()
        axn[i, j].set_title(title)
        counter += 1


fig, axn = plt.subplots(3, 4, sharex=True, sharey=False, figsize=(20,7))
images=[]
uniq_vals = np.unique(c.direct_samples.T[:,450:500])
minv, maxv = uniq_vals[0], uniq_vals[-1]
studies = ["S1", "S2", "S3"]
conditions = [["Control", "Altitude", "Temp", "Altitude + Temp"], 
              ["Control", "Light", "Temp", "Light + Temp"],
              ["Control", "Food", "Dirt", "Food + Dirt"]]
for i in range(3):
    for j in range(4):
        if i == 0:
            minn = 0
            maxn = 80
        if i == 1:
            minn = 80
            maxn = 144
        if i == 2:
            minn = 144
            maxn = 184
        title = studies[i] + ": " + conditions[i][j]
        data = c.direct_samples.T[minn+j:maxn:4,450:500]
        images.append(axn[i,j].imshow(data, cmap="jet", vmin=minv, vmax=maxv))
        axn[i, j].label_outer()
        axn[i, j].set_title(title)