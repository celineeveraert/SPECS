#!/usr/bin/env python3
import itertools
import pandas as pd
import numpy as np
import scipy.stats as sc
from sys import argv
import random
import math

def rankdata(a, method='average'):
    arr = np.ravel(np.asarray(a))
    algo = 'mergesort' if method == 'ordinal' else 'quicksort'
    sorter = np.argsort(arr, kind=algo)
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)

def mannwhitneyu(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1*n2 + (n1*(n1+1))/2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1*n2 - u1  # remainder is U for y
    return(u2)

def variance_pi(x1,x2):
    n1 = len(x1)
    n2 = len(x2)
    wc_stat = mannwhitneyu(x1,x2)
    pi = wc_stat/(n1*n2)
    return(pi)

#reading file
infile = open(argv[1], 'r')
samples = tuple(infile.readline().rstrip().split('\t'))
samples = samples[1:]

#determine unique cancertypes
diseases = tuple([sample.split('_',1)[0] for sample in samples])
diseases = np.array(diseases)

table = pd.Series(diseases)
counts = table.value_counts(sort=False)
tot_dis=len(table)
dis = len(counts)

unique_dis= []
seen_dis = set()
for i in diseases:
    if i not in seen_dis:
        unique_dis.append(i)
        seen_dis.add(i)
print(unique_dis)

#calculate weight factor, in this case equally weighted
pk=1/(len(unique_dis)-1)

#read names transcripts
infile2 = open(argv[2], 'r')
rnas = infile2.read().split('\n')
rnas = list(filter(None, rnas[1:]))

mw_RNA = np.zeros((len(rnas), dis))
s_RNA = np.zeros((len(rnas), dis))
transcript = infile.readline()
i=0

#calculation of score for each transcript
while transcript:
    tmp_RNA= transcript.split('\t')
    tmp_RNA = tmp_RNA[1:]
    tmp_RNA= np.array([float(value) for value in tmp_RNA])
    j=0
    #run through next loop for each cancer type (d)
    for sel_dis in unique_dis:
        #print(sel_dis)
        tmp_RNA_selected = tmp_RNA[diseases == sel_dis]
        #print(tmp_RNA_selected)
        unique_dis_non = [unique_dis[k] for k in range(len(unique_dis)) if unique_dis[k] != sel_dis]
        ptot_dis = 0
        var_list = []
        p_list = []
        #compare with each cancer type not d (k)
        cov_matrix = np.zeros(shape=(len(unique_dis_non),len(unique_dis_non)))
        k=0
        for comp_dis in unique_dis_non:
            #print(comp_dis)
            l=0
            tmp_RNA_comp = tmp_RNA[diseases == comp_dis]
            wc_stat_kd = variance_pi(tmp_RNA_selected,tmp_RNA_comp)
            #(wc_stat_kd,var_pi_kd) = variance_pi(tmp_RNA_selected,tmp_RNA_comp)
            #print(var_pi_kd)
            ptot_dis = ptot_dis + wc_stat_kd * pk
            #cov_matrix[k,k] = var_pi_kd
            p_list.append(wc_stat_kd)
        mw_RNA[i,j] = ptot_dis
        j += 1
    transcript = infile.readline()
    i += 1

#writing output
mw_RNA = pd.DataFrame(mw_RNA, index=rnas, columns=unique_dis)
mw_RNA.to_csv(argv[3]+"_pall_out.txt", sep='\t', index=True, header=True)

mw_max_type = pd.concat([mw_RNA.max(axis=1),mw_RNA.idxmax(axis=1)], axis=1)
mw_max_type = mw_max_type.sort_values(by=[0,1], axis=0, ascending=False)
mw_max_type.to_csv(argv[3]+"_onco_out.txt", sep='\t', index=True, header=False)

mw_min_type = pd.concat([mw_RNA.min(axis=1),mw_RNA.idxmin(axis=1)], axis=1)
mw_min_type = mw_min_type.sort_values(by=[0,1], axis=0, ascending=False)
mw_min_type.to_csv(argv[3]+"_tumsup_out.txt", sep='\t', index=True, header=False)