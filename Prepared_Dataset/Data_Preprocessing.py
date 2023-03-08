#!/usr/bin/env python
# coding: utf-8

# Enlarge Function

# In[ ]:


import numpy as np
from pandas import read_csv
import pandas as pd
import glob
import csv
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from numpy.lib import stride_tricks
from itertools import zip_longest

filenames = glob.glob("Letters/*.csv")

#filenames = ["AA/P2.N.W2().csv"]

file = []
for filename in filenames:
    filename = str(filename)
    first = pd.read_csv(filename,header = None)
    ff = np.array(first)
    first1 = np.shape(ff)
    
    if (first1 < (66,)):
            features = np.array(first).transpose()
            f = features.transpose()

            ma = MaxAbsScaler()
            a = ma.fit_transform(features).transpose()
            D1 = a[0]
            D2 = a[1]
            avg1 = (D1[1:] + D1[:-1]) / 2
            avg2 = (D2[1:] + D2[:-1]) / 2
            AVG1 = np.append(avg1 , avg1[0])
            AVG2 = np.append(avg2 , avg2[0])
    
            AA = 66 - len(D1)
            b = AVG1[:AA]
            c = AVG2[:AA]

            z = list(zip_longest(D1, b))
            ZZ = np.array(z).flatten()
            ZZ = [x for x in ZZ if x is not None]
            Z1 = np.array(ZZ).flatten()


            y = list(zip_longest(D2, c))
            yy = np.array(y).flatten()
            yy = [x for x in yy if x is not None]
            Z2 = np.array(yy).flatten()
    
            Main = np.array([Z1,Z2])
    
     
            with open(str(filename),"w+") as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(Main)


# In[ ]:





# Reduce Function

# In[ ]:


from pandas import read_csv
import pandas as pd
import io
import glob
import os
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import csv
from scipy.signal import resample
from more_itertools import unique_everseen


filenames = glob.glob("Letters/*.csv")


file = []
for filename in filenames:
    filename = str(filename)
    first = pd.read_csv(filename,header = None)
    ff = np.array(first).flatten()
    first1 = np.shape(ff)
    
    if (first1 > (66,)):
            features = np.array(first).transpose()
            f = features.transpose()

            ma = MaxAbsScaler()
            a = ma.fit_transform(features)
            feature = resample(a, 78)
            feature  =  np.array(feature).transpose()
            x = feature[0]
            y = feature[1]
            Z = feature
            Z[0]
            ZZ = Z[0][12:]
            ZZ = ZZ[:-12]
            ZZZ = Z[1][12:]
            ZZZ = ZZZ[:-12]

            Main2 = np.array([ZZ,ZZZ])
            
            with open(str(filename),"w+") as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(Main2)


# In[ ]:





# Padding

# In[ ]:


from pandas import read_csv
import pandas as pd
import io
import glob
import os
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import csv
from scipy.signal import resample
from more_itertools import unique_everseen

filenames = sorted(glob.glob("Writing/Feature_9/W_W/*.csv"))


file = []
for filename in filenames:
    first = pd.read_csv(filename)
    DATA = np.array(first)
    file.append(DATA)


# In[ ]:


F = np.array(file)
file[9780].shape


# In[ ]:


sizes = []
F = np.array(file)

for i in F:
    A = len(i)
    sizes.append(A)
    
max(sizes)


#Number of sequences
sizes = np.array(sizes)
for idx, i in enumerate(sizes):
    if (i >= 598):
        print(idx, i)


# In[ ]:


from keras.preprocessing.sequence import pad_sequences

F = np.array(file)

XX = pad_sequences(F,dtype=object, maxlen=599)


# In[ ]:


np.save('Feature_9_W.npy',XX)

