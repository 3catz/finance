import numpy as np 
import pyentrp as pe 

def incren(ts,m,R,fd = True):
  from pyentrp import entropy as ent 
  diff = np.diff(ts)
  subvectors = ent.util_pattern_space(diff, lag = 1, dim = m)
  signs = np.sign(subvectors)
  allqs = []  
  for i in range(len(subvectors)):
      sd = np.std(subvectors[i,:])
      row = subvectors[i,:]
      qs = []
      for element in row:
         q = R * np.abs(element)/sd   
         q = np.min([R,q])
         q = np.floor(q)
         qs.append(q)
      allqs.append(qs)
  allqs = np.vstack(allqs)
  words = np.multiply(signs, allqs)
  unq_rows, count = np.unique(words,axis=0, return_counts=1)
  out = {tuple(i):j for i,j in zip(unq_rows,count)}
  freq_dict = {}
  for key in out.keys():
    freq = {key:out[key]/len(words)}
    freq_dict.update(freq)
  ent = 0 
  for freq in freq_dict.values():
    ent += freq * np.log2(freq);
  if fd == True:
    return -1 * ent, freq_dict
  else:
    return -1 * ent, words
