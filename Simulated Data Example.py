# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:30:08 2020

@author: morri
"""
import numpy as np

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
Study1_rates = np.random.uniform(low=0, high=40, size=4)
Study2_rates = np.array((Study1_rates[0]+np.random.uniform(low=-1.5, high=1.5), 
                         np.random.uniform(low=0, high=40), 
                         Study1_rates[2]+np.random.uniform(low=-1.5, high=1.5), 
                         np.random.uniform(low=0, high=40)))
Study3_rates = np.random.uniform(low=0, high=40, size=4)
Study3_rates[0] = Study2_rates[0]+np.random.uniform(low=-1.5, high=1.5)


#Each set of conditions in study 1 done 20 times, study 2 16 times,
#study 3 10 times:
study1_obs = np.random.poisson(lam=Study1_rates, size=(20,4))
study2_obs = np.random.poisson(lam=Study2_rates, size=(16,4))
study3_obs = np.random.poisson(lam=Study3_rates, size=(10,4))

pop_obs = np.concatenate((study1_obs.flatten(), study2_obs.flatten(), study3_obs.flatten()))
study_tracker = np.repeat(np.array(["Study1", "Study2", "Study3"]), [20*4, 16*4, 10*4])
cond_tracker = np.concatenate(np.array((["Control", "Alt", "Temp", "AltTemp"]*20, 
                                        ["Control", "Sunlight", "Temp", "SunlightTemp"]*16, 
                                        ["Control", "Fertility", "Purity", "FertilityPurity"]*10)).flatten())
study_factor = np.unique(study_tracker, return_inverse=True)[1]
%time c = HDP(f='poisson', hypers=(6,1)).gibbs_direct(pop_obs[:,None], study_factor, iters=300, Kmax=12)
%time d = HDP(f='poisson', hypers=(6,1)).gibbs_cfr(pop_obs[:,None], study_factor, iters=300, Kmax=12)
