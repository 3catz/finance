#!pip install fix_yahoo_finance --upgrade --no-cache-dir
# !pip install saxpy
import pandas as pd
from tqdm import tqdm
import numpy as np
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize


df = []
for year in range(1994,2019):
   hist = yf.download(tickers = "SPY", 
                   start="{}-01-01".format(year), 
                   end = "{}-12-31".format(year) 
                     )
   close = hist["Close"]
   df.append(close)
   
 

words = []
for year in tqdm(df):
  
  #dat = ent.util_granulate_time_series(year, scale=3)
  dat_znorm = znorm(year)
  dat_paa= paa(dat_znorm, 10)
  word = ts_to_string(dat_paa, cuts_for_asize(5))
  words.append(word)
print(words)

from collections import Counter
Counter(words)

from collections import Counter
years = np.arange(1994,2019)
frame = pd.DataFrame()
frame["Year"]=years
frame["Word"]=words
print(frame)

from fuzzywuzzy import fuzz
all_ratios = []
for i in range(len(words)):
  year_ratios = []
  for j in range(len(words)):
    r = fuzz.ratio(words[i],words[j])
    year_ratios.append(r)
  all_ratios.append(year_ratios)
  
comps = pd.DataFrame(np.vstack(all_ratios))
comps.columns=np.arange(1994,2019)
comps.index = np.arange(1994,2019)
comps


for i in range(len(comps)):
  f = [x for x in comps.iloc[i,:] if x!=100]
  most = np.argmax(f)
  print((i+1994),":",most + 1994,str(np.max(f))+"% similarity score")
