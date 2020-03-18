import numpy as np
import sys
import collections

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
  


def Freq(to_check):
    to_check = word_tokenize(to_check)
    stop_words = set(stopwords.words('english')) 
    to_check = set(to_check) - stop_words
    arr = np.load('Frequency.npy')
    freq_dict = collections.OrderedDict()
    for i,x in enumerate(arr):
        temp = x.split()
        freq_dict[temp[0].upper()] = i
    ans = []
    for x in to_check:
        try:
            ans.append(freq_dict[x.upper()])
        except:
            ans.append(100000)

    return ans