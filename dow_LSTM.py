!pip install pyentrp
!pip install tslearn
!pip install git+https://github.com/manu-mannattil/nolitsa.git
!pip install saxpy
!pip install requests_html
!pip install fix_yahoo_finance --upgrade --no-cache-dir

from tslearn.clustering import TimeSeriesKMeans
import numpy as np 
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from nolitsa import noise, utils
import fix_yahoo_finance as yf 

from tqdm import tqdm
from pyentrp import entropy as ent
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
import collections 

import keras
from keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from keras.optimizers import Adam



##########

hist = yf.download(tickers = "DJI", period = 'max')



words = [] 
dow_df = ent.util_pattern_space(hist_sma, lag = 1, dim = 50)
dow_df = dow_df[:]
for i in range(len(dow_df)):
    dat_znorm = znorm(dow_df[i,:])
    dat_paa= paa(dat_znorm, 3)
    word = ts_to_string(dat_paa, cuts_for_asize(2))
    words.append(word)
print(words)


print(collections.Counter(words))

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sqn = le.fit_transform(words)

nb_classes = len(np.unique(sqn))

from keras.utils import to_categorical 
onehot = to_categorical(sqn)

print(nb_classes)
L = len(sqn)

win = 10
X = [] ; Y = [] 
for i in range(L - win + 1):
  x = onehot[i : i + win] ; y = onehot[win + 11]
  X.append(x); Y.append(y)
  
Y = np.array(Y)

################Keras#######################

early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, 
              patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.8, patience=5, verbose=1)


# define problem properties
n_timesteps = X.shape[1]
output = X.shape[-1]
# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(32, kernel_initializer = "he_normal",return_sequences=False),
                           input_shape=(n_timesteps, output)))

#model.add(Bidirectional(LSTM(32, return_sequences = False)))                     
model.add(Dense(output, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=.001), metrics=['acc'])


seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
model.fit(X, Y, epochs=1000, 
          validation_split = 0.2,
          batch_size=256, 
          callbacks = [early, reduce],
          shuffle = True,
          verbose=1)

preds = model.predict_classes(X)
Y_classes = [np.argmax(y) for y in Y]
Y_classes = np.array(Y_classes)
print(confusion_matrix(Y_classes, preds))
print(f1_score(Y_classes, preds, average='micro'))
















