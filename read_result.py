#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:23:01 2021

@author: simone
"""
#%%
import pickle
import numpy as np

model = 'PCA'
n_fold = 5
pr_aucs = []
roc_aucs = []
f2_scores = []

for i in range(n_fold):
    with open('models/'+model+"/fold_"+str(i)+"/result.pickle", 'rb') as f:
        result = pickle.load(f)
    print("\n \t fold "+str(i))
    print(result)
    pr_aucs.append(result['pr_auc'])
    roc_aucs.append(result['roc_auc'])
    f2_scores.append(result['f2score'])

print("\n pr auc: %.4f +- %.4f" % (np.mean(np.array(pr_aucs)), np.std(np.array(pr_aucs))))
print("\n roc auc: %.4f +- %.4f" % (np.mean(np.array(roc_aucs)), np.std(np.array(roc_aucs))))
print("\n f2 score: %.4f +- %.4f" % (np.mean(np.array(f2_scores)), np.std(np.array(f2_scores))))

#%%
with open("/home/simone/Downloads/result.pickle", 'rb') as f:
    result = pickle.load(f)
print(result)
